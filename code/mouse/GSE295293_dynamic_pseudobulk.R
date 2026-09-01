suppressPackageStartupMessages({
  library(data.table)
  library(edgeR)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  index <- match(flag, args)
  if (!is.na(index) && index < length(args)) args[[index + 1L]] else default
}
repo_root <- Sys.getenv("PATHONICHE_REPO", ".")
input_dir <- arg_value("--input-dir", file.path(repo_root, "work/GSE295293"))
output_dir <- arg_value("--output-dir", file.path(repo_root, "results/GSE295293_dynamic_regression"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

counts_file <- file.path(input_dir, "pseudobulk_counts.tsv.gz")
counts <- fread(cmd = paste("gzip -cd", shQuote(counts_file)))
genes <- counts[[1L]]
counts <- as.matrix(counts[, -1L])
rownames(counts) <- genes
metadata <- fread(file.path(input_dir, "pseudobulk_metadata.tsv"))
metadata[, condition := factor(condition, levels = c("control", "fibrosis", "regression"))]

cosine <- function(x, y) sum(x * y, na.rm = TRUE) /
  sqrt(sum(x^2, na.rm = TRUE) * sum(y^2, na.rm = TRUE))

summary_rows <- list()
all_results <- list()

for (ct in sort(unique(metadata$cell_type))) {
  meta <- metadata[cell_type == ct]
  if (any(table(meta$condition) < 2)) next
  matrix <- counts[, meta$pseudobulk, drop = FALSE]
  y <- DGEList(matrix)
  design <- model.matrix(~ 0 + condition, data = meta)
  colnames(design) <- sub("condition", "", colnames(design))
  keep <- filterByExpr(y, design)
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- calcNormFactors(y)
  y <- estimateDisp(y, design, robust = TRUE)
  fit <- glmQLFit(y, design, robust = TRUE)
  contrasts <- list(
    progression = makeContrasts(fibrosis - control, levels = design),
    regression = makeContrasts(regression - fibrosis, levels = design),
    residual = makeContrasts(regression - control, levels = design)
  )
  tables <- lapply(names(contrasts), function(name) {
    result <- topTags(glmQLFTest(fit, contrast = contrasts[[name]]), n = Inf, sort.by = "none")$table
    data.table(gene = rownames(result), contrast = name, result)
  })
  names(tables) <- names(contrasts)
  combined <- rbindlist(tables)
  fwrite(combined, file.path(output_dir, paste0("edgeR_", ct, ".tsv.gz")), sep = "\t")

  progression <- tables$progression[, .(gene, progression_logFC = logFC, progression_FDR = FDR)]
  regression <- tables$regression[, .(gene, regression_logFC = logFC, regression_FDR = FDR)]
  residual <- tables$residual[, .(gene, residual_logFC = logFC, residual_FDR = FDR)]
  joint <- Reduce(function(x, z) merge(x, z, by = "gene"), list(progression, regression, residual))
  disease <- joint$progression_FDR < 0.10 & abs(joint$progression_logFC) >= 0.5
  reversed <- disease & sign(joint$regression_logFC) == -sign(joint$progression_logFC) &
    abs(joint$regression_logFC) >= 0.5 * abs(joint$progression_logFC)
  persistent <- disease & sign(joint$residual_logFC) == sign(joint$progression_logFC) &
    joint$residual_FDR < 0.10 & abs(joint$residual_logFC) >= 0.5 * abs(joint$progression_logFC)
  joint[, fate := fifelse(reversed, "reversed",
                          fifelse(persistent, "persistent", "partial_or_other"))]
  fwrite(joint[order(progression_FDR)], file.path(output_dir, paste0("gene_fates_", ct, ".tsv.gz")), sep = "\t")

  summary_rows[[ct]] <- data.table(
    cell_type = ct,
    n_samples = nrow(meta),
    n_genes = nrow(joint),
    progression_FDR05 = sum(joint$progression_FDR < 0.05),
    regression_FDR05 = sum(joint$regression_FDR < 0.05),
    residual_FDR05 = sum(joint$residual_FDR < 0.05),
    disease_genes_FDR10_FC05 = sum(disease),
    reversed_genes = sum(reversed),
    persistent_genes = sum(persistent),
    recovery_fraction = ifelse(sum(disease) > 0, sum(reversed) / sum(disease), NA_real_),
    persistence_fraction = ifelse(sum(disease) > 0, sum(persistent) / sum(disease), NA_real_),
    regression_vs_negative_progression_spearman = cor(joint$regression_logFC,
                                                        -joint$progression_logFC,
                                                        method = "spearman"),
    regression_vs_negative_progression_cosine = cosine(joint$regression_logFC,
                                                         -joint$progression_logFC),
    nonreciprocity_norm = sqrt(sum((joint$progression_logFC + joint$regression_logFC)^2)) /
      (sqrt(sum(joint$progression_logFC^2)) + sqrt(sum(joint$regression_logFC^2)))
  )
}

summary <- rbindlist(summary_rows, fill = TRUE)
fwrite(summary, file.path(output_dir, "celltype_dynamic_summary.tsv"), sep = "\t")
write_json(list(
  analysis = "donor-level pseudobulk progression-regression geometry",
  statistical_unit = "mouse",
  progression = "fibrosis minus control",
  regression = "regression minus fibrosis",
  residual = "regression minus control",
  caveat = "nonreciprocity requires independent replication and a label-aware null; cells are not replicates",
  cell_types = nrow(summary)
), file.path(output_dir, "dynamic_analysis_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
print(summary)
