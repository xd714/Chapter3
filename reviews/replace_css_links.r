# R Script to Replace CSS Links in HTML Files
# This script replaces the old style.css link with the new modular CSS files

library(stringr)

# Define the old CSS link pattern to find
new_css_links <- '<link rel="stylesheet" href="../style/main.css">'

# Define the new CSS links to replace with
old_css_pattern <- '<link rel="stylesheet" href="style.css">'

# Function to process a single HTML file
process_html_file <- function(file_path) {
  # Read the file
  content <- readLines(file_path, warn = FALSE)
  content <- paste(content, collapse = "\n")
  
  # Check if the old CSS link exists
  if (str_detect(content, fixed(old_css_pattern))) {
    # Replace the old CSS link with new ones
    new_content <- str_replace(content, fixed(old_css_pattern), new_css_links)
    
    # Write the updated content back to file
    writeLines(strsplit(new_content, "\n")[[1]], file_path)
    
    cat("✅ Updated:", file_path, "\n")
    return(TRUE)
  } else {
    cat("⚠️  No old CSS link found in:", file_path, "\n")
    return(FALSE)
  }
}

# Get all HTML files in current directory
html_files <- list.files(pattern = "\\.html$", full.names = TRUE)

if (length(html_files) == 0) {
  cat("❌ No HTML files found in current directory.\n")
  cat("💡 Make sure you're in the correct directory with HTML files.\n")
} else {
  cat("🔍 Found", length(html_files), "HTML files:\n")
  cat(paste("  -", basename(html_files)), sep = "\n")
  cat("\n")
  
  # Process each HTML file
  cat("🔄 Processing files...\n\n")
  updated_count <- 0
  
  for (file in html_files) {
    if (process_html_file(file)) {
      updated_count <- updated_count + 1
    }
  }
  
  cat("\n✨ Process completed!\n")
  cat("📊 Summary:", updated_count, "out of", length(html_files), "files updated.\n")
  
  if (updated_count > 0) {
    cat("\n📝 Next steps:\n")
    cat("1. Make sure all 6 CSS files are in the same directory as your HTML files\n")
    cat("2. Test your HTML files in a browser to ensure everything works\n")
    cat("3. The new CSS system includes 30 automatic color themes!\n")
  }
}

# Optional: Also search in subdirectories
cat("\n🔍 Do you want to search subdirectories too? (y/n): ")
answer <- tolower(trimws(readline()))

if (answer == "y" || answer == "yes") {
  # Get all HTML files recursively
  all_html_files <- list.files(pattern = "\\.html$", recursive = TRUE, full.names = TRUE)
  subdirectory_files <- setdiff(all_html_files, html_files)
  
  if (length(subdirectory_files) > 0) {
    cat("\n🔍 Found", length(subdirectory_files), "additional HTML files in subdirectories:\n")
    cat(paste("  -", subdirectory_files), sep = "\n")
    cat("\n🔄 Processing subdirectory files...\n\n")
    
    subdirectory_updated <- 0
    for (file in subdirectory_files) {
      if (process_html_file(file)) {
        subdirectory_updated <- subdirectory_updated + 1
      }
    }
    
    cat("\n✨ Subdirectory processing completed!\n")
    cat("📊 Additional files updated:", subdirectory_updated, "\n")
    cat("📊 Total files updated:", updated_count + subdirectory_updated, "\n")
    
    if (subdirectory_updated > 0) {
      cat("\n⚠️  Note: Files in subdirectories will need their CSS paths adjusted\n")
      cat("   Example: href=\"../core_academics.css\" (if CSS files are in parent directory)\n")
    }
  } else {
    cat("✅ No additional HTML files found in subdirectories.\n")
  }
}