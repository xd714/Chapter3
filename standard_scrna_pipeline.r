# =============================================================================
# STANDARD SINGLE-CELL RNA-SEQ ANALYSIS PIPELINE
# =============================================================================
# Author: XD
# Date: 29.06.2025
# Description: Comprehensive pipeline for scRNA-seq analysis
# Compatible with: 10X Genomics data, Bioconductor & Seurat workflows
# =============================================================================

# =============================================================================
# 1. SETUP AND CONFIGURATION
# =============================================================================

# Clear environment
rm(list = ls())
gc()

# Load required libraries
suppressPackageStartupMessages({
  # Core analysis
  library(Seurat)               # Main scRNA-seq analysis
  library(SingleCellExperiment) # Bioconductor scRNA-seq
  library(scater)               # QC and visualization
  library(scran)                # Normalization and clustering
  library(DropletUtils)         # 10X data import
  library(sctransform)          # Advanced normalization
  
  # Annotation
  library(AnnotationHub)        # Gene annotations
  library(ensembldb)            # Ensembl annotations
  library(biomaRt)              # Alternative annotations
  
  # Data manipulation
  library(dplyr)                # Data wrangling
  library(tibble)               # Enhanced data frames
  library(purrr)                # Functional programming
  
  # Visualization
  library(ggplot2)              # Plotting
  library(patchwork)            # Combine plots
  library(viridis)              # Color schemes
  library(RColorBrewer)         # Additional colors
  library(pheatmap)             # Heatmaps
  library(ComplexHeatmap)       # Advanced heatmaps
  
  # Clustering and analysis
  library(bluster)              # Clustering methods
  library(cluster)              # Clustering validation
  library(igraph)               # Network analysis
  
  # Utilities
  library(Matrix)               # Sparse matrices
  library(scales)               # Scale functions
  library(gridExtra)            # Grid layouts
})

# =============================================================================
# 2. PROJECT CONFIGURATION
# =============================================================================

# PROJECT PARAMETERS - MODIFY THESE FOR YOUR PROJECT
config <- list(
  # Project info
  project_name = "MyProject",
  sample_name = "Sample1",
  species = "Homo sapiens",  # or "Mus musculus", "Bos taurus", etc.
  
  # Data paths
  data_dir = "data/",                    # 10X data directory
  output_dir = "results/",               # Output directory
  
  # Analysis parameters
  workflow = "seurat",                   # "seurat" or "bioconductor"
  min_cells = 3,                         # Min cells per gene
  min_features = 200,                    # Min genes per cell
  
  # QC thresholds
  max_mito_percent = 20,                 # Max mitochondrial %
  min_genes = 200,                       # Min genes per cell
  max_genes = 6000,                      # Max genes per cell
  min_umi = 500,                         # Min UMI per cell
  max_umi = 50000,                       # Max UMI per cell
  min_complexity = 0.8,                  # Min log10(genes)/log10(UMI)
  
  # Normalization
  normalization_method = "LogNormalize", # "LogNormalize", "sctransform"
  scale_factor = 10000,                  # Scaling factor
  
  # Feature selection
  n_variable_features = 2000,            # Number of variable features
  
  # Dimensionality reduction
  n_pcs = 50,                           # Number of PCs to compute
  n_pcs_use = 20,                       # Number of PCs to use
  
  # Clustering
  cluster_resolution = 0.5,              # Clustering resolution
  cluster_algorithm = 1,                 # 1=Louvain, 2=Louvain+multilevel, 3=SLM
  
  # Visualization
  umap_n_neighbors = 30,                # UMAP neighbors
  umap_min_dist = 0.3,                  # UMAP min distance
  tsne_perplexity = 30                  # t-SNE perplexity
)

# Create output directories
dir.create(config$output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(config$output_dir, "plots"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(config$output_dir, "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(config$output_dir, "objects"), recursive = TRUE, showWarnings = FALSE)

# Set random seed for reproducibility
set.seed(42)

# =============================================================================
# 3. UTILITY FUNCTIONS
# =============================================================================

# Function to save plots
save_plot <- function(plot, filename, width = 10, height = 8, dpi = 300) {
  ggsave(
    filename = file.path(config$output_dir, "plots", filename),
    plot = plot,
    width = width,
    height = height,
    dpi = dpi
  )
}

# Function to get species-specific annotation
get_species_annotation <- function(species) {
  species_map <- list(
    "Homo sapiens" = list(pattern = "^MT-", ensdb_query = c("Homo sapiens", "EnsDb")),
    "Mus musculus" = list(pattern = "^mt-", ensdb_query = c("Mus musculus", "EnsDb")),
    "Bos taurus" = list(pattern = "^MT", ensdb_query = c("Bos taurus", "EnsDb"))
  )
  return(species_map[[species]])
}

# Function to calculate QC metrics
calculate_qc_metrics <- function(object, species_info) {
  if (class(object)[1] == "Seurat") {
    # Seurat object
    object$percent_mito <- PercentageFeatureSet(object, pattern = species_info$pattern)
    object$percent_ribo <- PercentageFeatureSet(object, pattern = "^RP[SL]")
    object$nCount_RNA_log10 <- log10(object$nCount_RNA)
    object$nFeature_RNA_log10 <- log10(object$nFeature_RNA)
    object$complexity <- log10(object$nFeature_RNA) / log10(object$nCount_RNA)
  } else {
    # SingleCellExperiment object
    # Implementation for SCE objects
    mito_genes <- grep(species_info$pattern, rownames(object), value = TRUE)
    ribo_genes <- grep("^RP[SL]", rownames(object), value = TRUE)
    
    object <- addPerCellQC(object, 
                          subsets = list(Mito = mito_genes, Ribo = ribo_genes))
  }
  return(object)
}

# Function to create QC plots
create_qc_plots <- function(object) {
  if (class(object)[1] == "Seurat") {
    # Seurat QC plots
    p1 <- VlnPlot(object, features = c("nFeature_RNA", "nCount_RNA", "percent_mito"), 
                  ncol = 3, pt.size = 0.1)
    
    p2 <- FeatureScatter(object, feature1 = "nCount_RNA", feature2 = "percent_mito") +
      geom_hline(yintercept = config$max_mito_percent, linetype = "dashed", color = "red")
    
    p3 <- FeatureScatter(object, feature1 = "nCount_RNA", feature2 = "nFeature_RNA") +
      geom_hline(yintercept = config$min_genes, linetype = "dashed", color = "red") +
      geom_hline(yintercept = config$max_genes, linetype = "dashed", color = "red")
    
    combined_plot <- p1 / (p2 | p3)
  } else {
    # SCE QC plots - implement as needed
    combined_plot <- plotColData(object, y = "subsets_Mito_percent")
  }
  
  return(combined_plot)
}

# =============================================================================
# 4. DATA LOADING
# =============================================================================

cat("Loading data...\n")

if (config$workflow == "seurat") {
  # Seurat workflow
  data <- Read10X(data.dir = config$data_dir)
  
  # Handle multi-sample data
  if (is.list(data)) {
    seurat_obj <- CreateSeuratObject(
      counts = data$`Gene Expression`,
      project = config$project_name,
      min.cells = config$min_cells,
      min.features = config$min_features
    )
  } else {
    seurat_obj <- CreateSeuratObject(
      counts = data,
      project = config$project_name,
      min.cells = config$min_cells,
      min.features = config$min_features
    )
  }
  
  # Add sample information
  seurat_obj$sample <- config$sample_name
  main_object <- seurat_obj
  
} else {
  # Bioconductor workflow
  sce_obj <- read10xCounts(config$data_dir, col.names = TRUE)
  
  # Filter genes and cells
  keep_genes <- rowSums(counts(sce_obj)) >= config$min_cells
  keep_cells <- colSums(counts(sce_obj)) >= config$min_features
  
  sce_obj <- sce_obj[keep_genes, keep_cells]
  main_object <- sce_obj
}

cat(sprintf("Loaded data: %d genes, %d cells\n", 
            nrow(main_object), ncol(main_object)))

# =============================================================================
# 5. GENE ANNOTATION
# =============================================================================

cat("Adding gene annotations...\n")

species_info <- get_species_annotation(config$species)

if (config$workflow == "seurat") {
  # Add gene metadata to Seurat object
  tryCatch({
    ah <- AnnotationHub()
    ensdb <- query(ah, species_info$ensdb_query)
    if (length(ensdb) > 0) {
      ensdb <- ensdb[[length(ensdb)]]  # Get latest version
      
      # Get gene annotations
      genes_df <- as.data.frame(genes(ensdb))
      
      # Map to features
      features <- rownames(main_object)
      main_object[["RNA"]][["symbol"]] <- features
      
      # Try to map chromosome info
      chr_map <- setNames(genes_df$seqnames, genes_df$gene_name)
      main_object[["RNA"]][["chromosome"]] <- chr_map[features]
    }
  }, error = function(e) {
    cat("Warning: Could not retrieve gene annotations\n")
  })
}

# =============================================================================
# 6. QUALITY CONTROL
# =============================================================================

cat("Calculating QC metrics...\n")

# Calculate QC metrics
main_object <- calculate_qc_metrics(main_object, species_info)

# Create and save QC plots before filtering
qc_plots_before <- create_qc_plots(main_object)
save_plot(qc_plots_before, "qc_metrics_before_filtering.png", width = 15, height = 10)

# Print QC summary before filtering
if (config$workflow == "seurat") {
  cat("Before filtering:\n")
  cat(sprintf("  Cells: %d\n", ncol(main_object)))
  cat(sprintf("  Genes: %d\n", nrow(main_object)))
  cat(sprintf("  Median genes per cell: %.0f\n", median(main_object$nFeature_RNA)))
  cat(sprintf("  Median UMI per cell: %.0f\n", median(main_object$nCount_RNA)))
  cat(sprintf("  Median mito %%: %.1f\n", median(main_object$percent_mito)))
}

# Apply QC filtering
if (config$workflow == "seurat") {
  main_object <- subset(main_object, subset = 
    nFeature_RNA > config$min_genes & 
    nFeature_RNA < config$max_genes & 
    nCount_RNA > config$min_umi & 
    nCount_RNA < config$max_umi &
    percent_mito < config$max_mito_percent &
    complexity > config$min_complexity
  )
} else {
  # Implement SCE filtering
  # Use adaptive thresholds or similar filtering logic
}

# Create and save QC plots after filtering
qc_plots_after <- create_qc_plots(main_object)
save_plot(qc_plots_after, "qc_metrics_after_filtering.png", width = 15, height = 10)

# Print QC summary after filtering
if (config$workflow == "seurat") {
  cat("After filtering:\n")
  cat(sprintf("  Cells: %d\n", ncol(main_object)))
  cat(sprintf("  Genes: %d\n", nrow(main_object)))
  cat(sprintf("  Median genes per cell: %.0f\n", median(main_object$nFeature_RNA)))
  cat(sprintf("  Median UMI per cell: %.0f\n", median(main_object$nCount_RNA)))
  cat(sprintf("  Median mito %%: %.1f\n", median(main_object$percent_mito)))
}

# =============================================================================
# 7. NORMALIZATION
# =============================================================================

cat("Normalizing data...\n")

if (config$workflow == "seurat") {
  if (config$normalization_method == "LogNormalize") {
    main_object <- NormalizeData(main_object, 
                                normalization.method = "LogNormalize",
                                scale.factor = config$scale_factor)
  } else if (config$normalization_method == "sctransform") {
    main_object <- SCTransform(main_object, 
                              vars.to.regress = c("percent_mito", "nCount_RNA"),
                              verbose = FALSE)
  }
} else {
  # Implement Bioconductor normalization
  # Use deconvolution or sctransform
}

# =============================================================================
# 8. FEATURE SELECTION
# =============================================================================

cat("Finding variable features...\n")

if (config$workflow == "seurat") {
  if (config$normalization_method != "sctransform") {
    main_object <- FindVariableFeatures(main_object, 
                                       selection.method = "vst",
                                       nfeatures = config$n_variable_features)
    
    # Plot variable features
    top10 <- head(VariableFeatures(main_object), 10)
    plot1 <- VariableFeaturePlot(main_object)
    plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
    
    var_features_plot <- plot1 | plot2
    save_plot(var_features_plot, "variable_features.png", width = 15, height = 6)
  }
}

# =============================================================================
# 9. SCALING AND PCA
# =============================================================================

cat("Scaling data and running PCA...\n")

if (config$workflow == "seurat") {
  if (config$normalization_method != "sctransform") {
    # Scale data
    all_genes <- rownames(main_object)
    main_object <- ScaleData(main_object, 
                            features = all_genes,
                            vars.to.regress = c("percent_mito", "nCount_RNA"))
  }
  
  # Run PCA
  main_object <- RunPCA(main_object, 
                       features = VariableFeatures(object = main_object),
                       npcs = config$n_pcs)
  
  # PCA plots
  pca_plot1 <- DimPlot(main_object, reduction = "pca")
  pca_plot2 <- ElbowPlot(main_object, ndims = 50)
  
  pca_combined <- pca_plot1 | pca_plot2
  save_plot(pca_combined, "pca_analysis.png", width = 15, height = 6)
  
  # PCA loadings
  pca_loadings <- VizDimLoadings(main_object, dims = 1:4, reduction = "pca", ncol = 2)
  save_plot(pca_loadings, "pca_loadings.png", width = 12, height = 8)
}

# =============================================================================
# 10. CLUSTERING
# =============================================================================

cat("Performing clustering...\n")

if (config$workflow == "seurat") {
  # Find neighbors
  main_object <- FindNeighbors(main_object, dims = 1:config$n_pcs_use)
  
  # Find clusters
  main_object <- FindClusters(main_object, 
                             resolution = config$cluster_resolution,
                             algorithm = config$cluster_algorithm)
  
  cat(sprintf("Found %d clusters\n", length(unique(Idents(main_object)))))
}

# =============================================================================
# 11. NON-LINEAR DIMENSIONALITY REDUCTION
# =============================================================================

cat("Running UMAP and t-SNE...\n")

if (config$workflow == "seurat") {
  # Run UMAP
  main_object <- RunUMAP(main_object, 
                        dims = 1:config$n_pcs_use,
                        n.neighbors = config$umap_n_neighbors,
                        min.dist = config$umap_min_dist)
  
  # Run t-SNE
  main_object <- RunTSNE(main_object, 
                        dims = 1:config$n_pcs_use,
                        perplexity = config$tsne_perplexity)
  
  # Create visualization plots
  umap_plot <- DimPlot(main_object, reduction = "umap", label = TRUE, pt.size = 0.5) +
    ggtitle("UMAP Clustering")
  
  tsne_plot <- DimPlot(main_object, reduction = "tsne", label = TRUE, pt.size = 0.5) +
    ggtitle("t-SNE Clustering")
  
  clustering_plots <- umap_plot | tsne_plot
  save_plot(clustering_plots, "clustering_umap_tsne.png", width = 16, height = 7)
}

# =============================================================================
# 12. MARKER GENE IDENTIFICATION
# =============================================================================

cat("Finding marker genes...\n")

if (config$workflow == "seurat") {
  # Find all markers
  all_markers <- FindAllMarkers(main_object, 
                               only.pos = TRUE,
                               min.pct = 0.25,
                               logfc.threshold = 0.25,
                               test.use = "wilcox")
  
  # Save marker genes
  write.csv(all_markers, 
           file.path(config$output_dir, "tables", "all_cluster_markers.csv"),
           row.names = FALSE)
  
  # Get top markers per cluster
  top_markers <- all_markers %>%
    group_by(cluster) %>%
    top_n(n = 5, wt = avg_log2FC) %>%
    ungroup()
  
  # Create heatmap of top markers
  if (nrow(top_markers) > 0) {
    heatmap_plot <- DoHeatmap(main_object, 
                             features = top_markers$gene, 
                             size = 3) +
      theme(axis.text.y = element_text(size = 8))
    
    save_plot(heatmap_plot, "marker_genes_heatmap.png", width = 12, height = 15)
  }
  
  # Create dot plot of top markers
  if (nrow(top_markers) > 0) {
    dot_plot <- DotPlot(main_object, features = unique(top_markers$gene)) +
      RotatedAxis() +
      theme(axis.text.x = element_text(size = 8))
    
    save_plot(dot_plot, "marker_genes_dotplot.png", width = 15, height = 8)
  }
}

# =============================================================================
# 13. CELL TYPE ANNOTATION (TEMPLATE)
# =============================================================================

cat("Cell type annotation (modify as needed)...\n")

# Define common marker genes (modify based on your tissue/species)
marker_genes <- list(
  "T_cells" = c("CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B"),
  "B_cells" = c("CD19", "MS4A1", "CD79A", "CD79B"),
  "NK_cells" = c("KLRD1", "KLRF1", "KLRB1", "NCR1"),
  "Monocytes" = c("CD14", "FCGR3A", "LYZ", "S100A8", "S100A9"),
  "Macrophages" = c("CD68", "CD163", "MRC1", "MARCO"),
  "Dendritic_cells" = c("FCER1A", "CST3", "CLEC9A"),
  "Epithelial_cells" = c("EPCAM", "KRT18", "KRT19"),
  "Endothelial_cells" = c("PECAM1", "VWF", "CDH5"),
  "Fibroblasts" = c("COL1A1", "COL3A1", "DCN", "LUM")
)

# Create feature plots for marker genes
for (cell_type in names(marker_genes)) {
  genes_present <- intersect(marker_genes[[cell_type]], rownames(main_object))
  
  if (length(genes_present) > 0) {
    if (config$workflow == "seurat") {
      feature_plot <- FeaturePlot(main_object, 
                                 features = genes_present[1:min(4, length(genes_present))],
                                 ncol = 2)
      
      save_plot(feature_plot, 
               paste0("markers_", cell_type, ".png"), 
               width = 12, height = 10)
    }
  }
}

# =============================================================================
# 14. SAVE RESULTS
# =============================================================================

cat("Saving results...\n")

# Save processed object
if (config$workflow == "seurat") {
  saveRDS(main_object, file.path(config$output_dir, "objects", "processed_seurat_object.rds"))
} else {
  saveRDS(main_object, file.path(config$output_dir, "objects", "processed_sce_object.rds"))
}

# Create summary statistics
summary_stats <- data.frame(
  Metric = c("Total_cells_before_QC", "Total_cells_after_QC", "Total_genes",
             "Number_of_clusters", "Median_genes_per_cell", "Median_UMI_per_cell"),
  Value = c(
    ifelse(exists("seurat_obj"), ncol(seurat_obj), "NA"),
    ncol(main_object),
    nrow(main_object),
    ifelse(config$workflow == "seurat", length(unique(Idents(main_object))), "NA"),
    ifelse(config$workflow == "seurat", median(main_object$nFeature_RNA), "NA"),
    ifelse(config$workflow == "seurat", median(main_object$nCount_RNA), "NA")
  )
)

write.csv(summary_stats, 
         file.path(config$output_dir, "tables", "analysis_summary.csv"),
         row.names = FALSE)

# Create final summary plot
if (config$workflow == "seurat") {
  # Cluster summary
  cluster_summary <- as.data.frame(table(Idents(main_object)))
  colnames(cluster_summary) <- c("Cluster", "Cell_Count")
  
  cluster_barplot <- ggplot(cluster_summary, aes(x = Cluster, y = Cell_Count)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(title = "Cells per Cluster", x = "Cluster", y = "Number of Cells") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Final UMAP with clusters
  final_umap <- DimPlot(main_object, reduction = "umap", label = TRUE, pt.size = 0.5) +
    ggtitle(paste("Final Clustering -", config$project_name)) +
    theme(legend.position = "right")
  
  final_summary <- final_umap | cluster_barplot
  save_plot(final_summary, "final_analysis_summary.png", width = 16, height = 7)
}

# =============================================================================
# 15. SESSION INFO AND COMPLETION
# =============================================================================

# Save session info
sink(file.path(config$output_dir, "session_info.txt"))
cat("Single-cell RNA-seq Analysis Pipeline\n")
cat("=====================================\n\n")
cat("Project:", config$project_name, "\n")
cat("Date:", Sys.Date(), "\n")
cat("Workflow:", config$workflow, "\n\n")
cat("Configuration:\n")
str(config)
cat("\n\nSession Info:\n")
sessionInfo()
sink()

cat("\n=============================================================================\n")
cat("ANALYSIS COMPLETE!\n")
cat("=============================================================================\n")
cat("Results saved to:", config$output_dir, "\n")
cat("- Plots: plots/\n")
cat("- Tables: tables/\n")
cat("- Objects: objects/\n")
cat("- Session info: session_info.txt\n")
cat("=============================================================================\n")

# =============================================================================
# 16. OPTIONAL: ADDITIONAL ANALYSES
# =============================================================================

# Uncomment and modify as needed for additional analyses

# # Differential expression between clusters
# if (config$workflow == "seurat") {
#   # Example: Compare cluster 1 vs all others
#   cluster1_markers <- FindMarkers(main_object, ident.1 = "1", min.pct = 0.25)
#   write.csv(cluster1_markers, 
#             file.path(config$output_dir, "tables", "cluster1_vs_all_markers.csv"))
# }

# # Cell cycle scoring
# if (config$workflow == "seurat" && config$species == "Homo sapiens") {
#   s.genes <- cc.genes$s.genes
#   g2m.genes <- cc.genes$g2m.genes
#   main_object <- CellCycleScoring(main_object, s.features = s.genes, 
#                                   g2m.features = g2m.genes, set.ident = TRUE)
#   
#   cc_plot <- DimPlot(main_object, reduction = "umap", group.by = "Phase")
#   save_plot(cc_plot, "cell_cycle_phases.png")
# }

# # Trajectory analysis (requires additional packages like monocle3 or slingshot)
# # Implement trajectory analysis if needed

# # Integration with reference datasets (requires additional packages)
# # Implement reference mapping if needed

# Print completion message
cat("Pipeline completed successfully!\n")
cat("To rerun with different parameters, modify the config list and rerun the script.\n")