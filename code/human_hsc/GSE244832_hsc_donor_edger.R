#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(data.table)
  library(edgeR)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  index <- match(flag, args)
  if (!is.na(index) && index < length(args)) args[[index + 1L]] else default
}
repo_root <- Sys.getenv("PATHONICHE_REPO", ".")
input <- arg_value("--input-dir", file.path(repo_root, "work/GSE244832_hsc_pseudobulk"))
out <- arg_value("--output-dir", file.path(repo_root, "results/GSE244832_hsc_donor"))
dir.create(out, recursive = TRUE, showWarnings = FALSE)

raw <- fread(file.path(input, "hsc_pseudobulk_counts.tsv"))
genes <- make.unique(raw$gene)
counts <- as.matrix(raw[, -1])
rownames(counts) <- genes
meta <- fread(file.path(input, "donor_metadata.tsv"))
stopifnot(identical(colnames(counts), paste0("donor_", meta$donor_index)))

label_sets <- list(metadata = meta$condition)
supplement <- meta$condition
supplement[meta$donor == "JB321"] <- "NAFL"
supplement[meta$donor == "JB337"] <- "NASH"
label_sets$supplement_swap <- supplement

all_summaries <- list()
for (label_name in names(label_sets)) {
  group <- factor(label_sets[[label_name]], levels = c("NORMAL", "NAFL", "NASH"))
  design <- model.matrix(~ 0 + group)
  colnames(design) <- levels(group)
  model_meta <- copy(meta)
  model_meta[, model_condition := as.character(group)]
  y <- DGEList(counts = counts, samples = model_meta)
  keep <- filterByExpr(y, group = group, min.count = 10, min.total.count = 30)
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- calcNormFactors(y)
  y <- estimateDisp(y, design, robust = TRUE)
  fit <- glmQLFit(y, design, robust = TRUE)

  contrasts <- list(
    MASH_vs_nonactivated = makeContrasts(NASH - (NORMAL + NAFL) / 2, levels = design),
    MASH_vs_NORMAL = makeContrasts(NASH - NORMAL, levels = design),
    MASL_vs_NORMAL = makeContrasts(NAFL - NORMAL, levels = design)
  )
  for (name in names(contrasts)) {
    test <- glmQLFTest(fit, contrast = contrasts[[name]])
    table <- topTags(test, n = Inf, sort.by = "PValue")$table
    table$gene <- rownames(table)
    fwrite(table, file.path(out, paste0(label_name, "_", name, "_edgeR.tsv")), sep = "\t")
    key <- paste(label_name, name, sep = "__")
    all_summaries[[key]] <- data.table(
      label_set = label_name,
      contrast = name,
      tested_genes = nrow(table),
      fdr_005 = sum(table$FDR < 0.05),
      fdr_010 = sum(table$FDR < 0.10),
      fdr_010_lfc_05 = sum(table$FDR < 0.10 & abs(table$logFC) >= 0.5)
    )
  }
  if (label_name == "metadata") {
    cpm_table <- as.data.table(cpm(y, log = TRUE, prior.count = 1), keep.rownames = "gene")
    fwrite(cpm_table, file.path(out, "hsc_logCPM.tsv.gz"), sep = "\t")
  }
  fwrite(as.data.table(y$samples), file.path(out, paste0(label_name, "_model_sample_metadata.tsv")), sep = "\t")
  saveRDS(list(y = y, fit = fit, design = design, contrasts = contrasts),
          file.path(out, paste0(label_name, "_edgeR_model.rds")), compress = "xz")
}

summary_table <- rbindlist(all_summaries)
fwrite(summary_table, file.path(out, "contrast_summary.tsv"), sep = "\t")
print(summary_table)
