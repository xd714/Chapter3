# R Script to Insert Viewport Meta Tag into HTML Files
# This script will add the viewport meta tag to all HTML files in a directory

# Load required libraries
library(stringr)

# Function to insert viewport meta tag
insert_viewport_meta <- function(file_path, backup = TRUE) {
  
  # Read the HTML file
  tryCatch({
    html_content <- readLines(file_path, warn = FALSE, encoding = "UTF-8")
    
    # Create backup if requested
    if (backup) {
      backup_path <- paste0(file_path, ".backup")
      writeLines(html_content, backup_path, useBytes = TRUE)
      cat("Backup created:", backup_path, "\n")
    }
    
    # Define the viewport meta tag to insert
    viewport_meta <- '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">'
    
    # Check if viewport meta tag already exists
    has_viewport <- any(str_detect(html_content, "viewport"))
    
    if (has_viewport) {
      cat("Viewport meta tag already exists in:", file_path, "\n")
      return(FALSE)
    }
    
    # Find the <head> tag
    head_line <- which(str_detect(html_content, "<head>|<head "))
    
    if (length(head_line) == 0) {
      cat("No <head> tag found in:", file_path, "\n")
      return(FALSE)
    }
    
    # Insert the viewport meta tag after the <head> tag
    head_line <- head_line[1]  # Use the first <head> tag if multiple exist
    
    # Create new content with viewport meta tag inserted
    new_content <- c(
      html_content[1:head_line],
      paste0("    ", viewport_meta),  # Add indentation
      html_content[(head_line + 1):length(html_content)]
    )
    
    # Write the modified content back to the file
    writeLines(new_content, file_path, useBytes = TRUE)
    
    cat("✓ Viewport meta tag added to:", file_path, "\n")
    return(TRUE)
    
  }, error = function(e) {
    cat("✗ Error processing", file_path, ":", e$message, "\n")
    return(FALSE)
  })
}

# Main function to process all HTML files
process_html_files <- function(directory_path = ".", pattern = "*.html", backup = TRUE) {
  
  # Validate directory
  if (!dir.exists(directory_path)) {
    stop("Directory does not exist: ", directory_path)
  }
  
  # Find all HTML files
  html_files <- list.files(
    path = directory_path, 
    pattern = "\\.html?$", 
    full.names = TRUE, 
    ignore.case = TRUE
  )
  
  if (length(html_files) == 0) {
    cat("No HTML files found in:", directory_path, "\n")
    return()
  }
  
  cat("Found", length(html_files), "HTML files\n")
  cat("Processing files...\n\n")
  
  # Process each file
  successful <- 0
  for (file in html_files) {
    if (insert_viewport_meta(file, backup)) {
      successful <- successful + 1
    }
  }
  
  cat("\n=== SUMMARY ===\n")
  cat("Total files processed:", length(html_files), "\n")
  cat("Successfully modified:", successful, "\n")
  cat("Skipped/failed:", length(html_files) - successful, "\n")
  
  if (backup) {
    cat("\nBackup files created with .backup extension\n")
  }
}

# Alternative function for more control
process_specific_files <- function(file_list, backup = TRUE) {
  
  cat("Processing", length(file_list), "specified files...\n\n")
  
  successful <- 0
  for (file in file_list) {
    if (file.exists(file)) {
      if (insert_viewport_meta(file, backup)) {
        successful <- successful + 1
      }
    } else {
      cat("✗ File not found:", file, "\n")
    }
  }
  
  cat("\n=== SUMMARY ===\n")
  cat("Total files processed:", length(file_list), "\n")
  cat("Successfully modified:", successful, "\n")
  cat("Skipped/failed:", length(file_list) - successful, "\n")
}

# Function to remove backup files (use with caution!)
remove_backups <- function(directory_path = ".") {
  backup_files <- list.files(
    path = directory_path, 
    pattern = "\\.backup$", 
    full.names = TRUE
  )
  
  if (length(backup_files) > 0) {
    cat("Found", length(backup_files), "backup files. Remove them? (y/n): ")
    response <- readline()
    
    if (tolower(response) %in% c("y", "yes")) {
      file.remove(backup_files)
      cat("Backup files removed.\n")
    } else {
      cat("Backup files kept.\n")
    }
  } else {
    cat("No backup files found.\n")
  }
}

# ===== USAGE EXAMPLES =====

# Example 1: Process all HTML files in current directory
# process_html_files()

# Example 2: Process all HTML files in a specific directory
# process_html_files("path/to/your/html/files")

# Example 3: Process specific files
# file_list <- c("file1.html", "file2.html", "file3.html")
# process_specific_files(file_list)

# Example 4: Process without creating backups (not recommended)
# process_html_files(backup = FALSE)

# ===== RUN THE SCRIPT =====

# Set your directory path here
html_directory <- "."  # Current directory, change this to your HTML files directory

# Run the main function
cat("=== HTML Viewport Meta Tag Inserter ===\n")
cat("This script will add viewport meta tags to your HTML files.\n")
cat("Backups will be created automatically.\n\n")

# Uncomment the line below to run:
process_html_files(html_directory, backup = TRUE)

# Uncomment to remove backup files after verification:
# remove_backups(html_directory)