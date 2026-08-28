#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(edgeR)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  index <- match(flag, args)
  if (!is.na(index) && index < length(args)) args[[index + 1L]] else default
}
repo_root <- Sys.getenv("PATHONICHE_REPO", ".")
source <- arg_value("--source-rds", file.path(repo_root, "data/raw_external/GSE261829/GSE261829_npc_filtered.rds.gz"))
out <- arg_value("--output-dir", file.path(repo_root, "results/GSE261829_macrophage_resolution"))
dir.create(out, recursive = TRUE, showWarnings = FALSE)

sample_map <- data.table(
  orig.ident = paste0("NPC_", c(3084, 3025, 2999, 2891, 2744, 2722, 2527, 2525, 1215)),
  condition = rep(c("control", "MASH", "regression"), each = 3),
  replicate = rep(1:3, 3)
)
sample_map[, condition := factor(condition, levels = c("control", "MASH", "regression"))]

object <- readRDS(gzcon(gzfile(source, "rb")))
meta <- as.data.table(attr(object, "meta.data"), keep.rownames = "cell")
counts <- attr(attr(object, "assays")[[1]], "counts")
stopifnot(identical(colnames(counts), meta$cell))
meta[, seurat_clusters := as.integer(as.character(seurat_clusters))]
meta <- merge(meta, sample_map, by = "orig.ident", all.x = TRUE, sort = FALSE)
stopifnot(!anyNA(meta$condition))

cluster_map <- data.table(
  seurat_clusters = c(0L, 8L, 1L, 2L, 5L, 15L),
  macrophage_state = c("EmKC", "EmKC", "MoKC", "LAM", "Ly6chi_RM", "Mac1")
)
meta <- merge(meta, cluster_map, by = "seurat_clusters", all.x = TRUE, sort = FALSE)
meta[, matrix_column := match(cell, colnames(counts))]

composition <- meta[!is.na(macrophage_state), .(n_cells = .N), by = .(orig.ident, macrophage_state)]
composition <- merge(
  CJ(orig.ident = sample_map$orig.ident, macrophage_state = unique(cluster_map$macrophage_state)),
  composition, by = c("orig.ident", "macrophage_state"), all.x = TRUE
)
composition[is.na(n_cells), n_cells := 0L]
composition <- merge(composition, sample_map, by = "orig.ident")
totals <- composition[, .(macrophage_cells = sum(n_cells)), by = .(orig.ident, condition, replicate)]
composition <- merge(composition, totals, by = c("orig.ident", "condition", "replicate"))
composition[, fraction := n_cells / macrophage_cells]
fwrite(composition, file.path(out, "donor_macrophage_composition.tsv"), sep = "\t")

aggregate_state <- function(state_name, clusters) {
  columns <- meta[seurat_clusters %in% clusters, matrix_column]
  state_meta <- meta[seurat_clusters %in% clusters, .(matrix_column, orig.ident)]
  result <- matrix(0, nrow = nrow(counts), ncol = nrow(sample_map),
                   dimnames = list(rownames(counts), sample_map$orig.ident))
  for (index in seq_len(nrow(sample_map))) {
    selected <- state_meta[orig.ident == sample_map$orig.ident[index], matrix_column]
    if (length(selected)) result[, index] <- Matrix::rowSums(counts[, selected, drop = FALSE])
  }
  result
}

run_dynamic <- function(state_name, clusters) {
  pseudobulk <- aggregate_state(state_name, clusters)
  group <- sample_map$condition
  design <- model.matrix(~ 0 + group)
  colnames(design) <- levels(group)
  y <- DGEList(pseudobulk, samples = sample_map)
  keep <- filterByExpr(y, group = group, min.count = 5, min.total.count = 15)
  y <- y[keep, , keep.lib.sizes = FALSE]
  y <- calcNormFactors(y)
  y <- estimateDisp(y, design, robust = TRUE)
  fit <- glmQLFit(y, design, robust = TRUE)
  contrasts <- list(
    progression = makeContrasts(MASH - control, levels = design),
    regression = makeContrasts(regression - MASH, levels = design),
    residual = makeContrasts(regression - control, levels = design)
  )
  tables <- list()
  for (name in names(contrasts)) {
    tab <- topTags(glmQLFTest(fit, contrast = contrasts[[name]]), n = Inf, sort.by = "none")$table
    tab$gene <- rownames(tab)
    tables[[name]] <- as.data.table(tab)
  }
  merged <- merge(tables$progression[, .(gene, progression_logFC = logFC, progression_FDR = FDR)],
                  tables$regression[, .(gene, regression_logFC = logFC, regression_FDR = FDR)], by = "gene")
  merged <- merge(merged, tables$residual[, .(gene, residual_logFC = logFC, residual_FDR = FDR)], by = "gene")
  merged[, fate := fifelse(
    progression_FDR < 0.10 & abs(progression_logFC) >= 0.5 &
      sign(residual_logFC) == sign(progression_logFC) & residual_FDR < 0.10 &
      abs(residual_logFC) >= 0.5 * abs(progression_logFC), "persistent",
    fifelse(progression_FDR < 0.10 & abs(progression_logFC) >= 0.5 &
              sign(regression_logFC) == -sign(progression_logFC) &
              abs(regression_logFC) >= 0.5 * abs(progression_logFC), "reversed", "partial_or_other")
  )]
  fwrite(merged, file.path(out, paste0("gene_fates_", state_name, ".tsv.gz")), sep = "\t")
  logcpm <- as.data.table(cpm(y, log = TRUE, prior.count = 1), keep.rownames = "gene")
  fwrite(logcpm, file.path(out, paste0("logCPM_", state_name, ".tsv.gz")), sep = "\t")
  list(y = y, fates = merged, logcpm = logcpm)
}

lam <- run_dynamic("LAM", 2L)
all_macrophages <- run_dynamic("all_macrophages", c(0L, 8L, 1L, 2L, 5L, 15L))

modules <- list(
  LAM_identity = c("Trem2", "Gpnmb", "Spp1", "Cd9", "Lpl", "Fabp5"),
  collagen_resolution = c("Mmp9", "Mmp12", "Mmp13", "Mmp14", "Plau", "Ctsb", "Ctsd", "Axl", "Mertk"),
  inflammatory = c("Il1b", "Tnf", "Nlrp3", "Ccl2", "Ccr2", "Ly6c2", "S100a8", "S100a9")
)
module_rows <- list()
for (state_name in c("LAM", "all_macrophages")) {
  table <- if (state_name == "LAM") lam$logcpm else all_macrophages$logcpm
  matrix_values <- as.matrix(table[, -1])
  rownames(matrix_values) <- table$gene
  for (module_name in names(modules)) {
    genes <- intersect(modules[[module_name]], rownames(matrix_values))
    score <- colMeans(matrix_values[genes, , drop = FALSE])
    module_rows[[paste(state_name, module_name)]] <- data.table(
      state = state_name, module = module_name, orig.ident = colnames(matrix_values), score = score
    )
  }
}
module_scores <- merge(rbindlist(module_rows), sample_map, by = "orig.ident")
fwrite(module_scores, file.path(out, "donor_module_scores.tsv"), sep = "\t")

lam_fraction <- composition[macrophage_state == "LAM"]
observed <- mean(lam_fraction[condition == "regression", fraction]) - mean(lam_fraction[condition == "MASH", fraction])
set.seed(20260711)
permuted <- replicate(10000, {
  shuffled <- sample(lam_fraction$condition)
  mean(lam_fraction$fraction[shuffled == "regression"]) - mean(lam_fraction$fraction[shuffled == "MASH"])
})
composition_test <- data.table(
  contrast = "LAM fraction: regression - MASH",
  effect = observed,
  permutation_p_two_sided = (1 + sum(abs(permuted) >= abs(observed))) / (length(permuted) + 1)
)
fwrite(composition_test, file.path(out, "LAM_composition_permutation.tsv"), sep = "\t")

priority <- c("Trem2", "Gpnmb", "Spp1", "Cd9", "Lpl", "Mmp9", "Mmp12", "Mmp13", "Mmp14", "Plau", "Axl", "Mertk")
priority_table <- lam$fates[gene %in% priority]
fwrite(priority_table, file.path(out, "LAM_priority_gene_fates.tsv"), sep = "\t")

print(composition)
print(composition_test)
print(priority_table)
