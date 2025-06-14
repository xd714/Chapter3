# R Script to Update Paper Links in HTML Review Files
# This script reads your HTML review files and replaces Google Scholar search links
# with actual paper URLs from your literature index

library(stringr)
library(xml2)

# Define the mapping of review files to actual paper URLs
paper_links <- list(
  "2009_chen.html" = "https://www.uni-goettingen.de/de/document/download/3032ef09f08ab4602f7a57f561ac9cd8.pdf/chen.pdf",
  "2009_Garrick.html" = "https://gsejournal.biomedcentral.com/articles/10.1186/1297-9686-41-55",
  "2022_palma_vera.html" = "https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-022-01248-9",
  "2025_niehoff.html" = "https://onlinelibrary.wiley.com/doi/full/10.1111/jbg.12913",
  "2006_bevova.html" = "https://academic.oup.com/genetics/article/172/1/401/6061240",
  "2025_rochus.html" = "https://doi.org/10.1101/2025.05.14.653764",
  "2019_stephan.html" = "https://www.genetics.org/content/211/1/5",
  "2018_marees.html" = "https://doi.org/10.1002/mpr.1608",
  "2017_xue.html" = "https://doi.org/10.1101/099291",
  "2018_oliveira.html" = "https://doi.org/10.1111/jbg.12317",
  "2025_musa.html" = "https://onlinelibrary.wiley.com/doi/full/10.1111/jbg.12930",
  "2008_vanraden.html" = "https://doi.org/10.3168/jds.2007-0980",
  "1991_vanraden.html" = "https://www.journalofdairyscience.org/article/S0022-0302(91)78463-X/fulltext",
  "2018_song.html" = "https://doi.org/10.1017/S175173111700307X",
  "2023_liu.html" = "https://www.nature.com/articles/s41467-023-41220-x"
)

# Function to find and display current links in a file
show_current_links <- function(file_path) {
  content <- readLines(file_path, warn = FALSE, encoding = "UTF-8")
  content_string <- paste(content, collapse = "\n")
  
  # Find all href links
  href_matches <- str_extract_all(content_string, 'href="[^"]*"')[[1]]
  if (length(href_matches) > 0) {
    cat("  Current links found:\n")
    for (link in href_matches) {
      cat("    ", link, "\n")
    }
  }
  
  # Find text that might indicate paper links
  paper_text_patterns <- c("Find Original Paper", "Google Scholar Search", "Search for Original Paper", "Original Paper", "View.*Paper")
  for (pattern in paper_text_patterns) {
    if (str_detect(content_string, pattern)) {
      cat("  Found text pattern:", pattern, "\n")
    }
  }
}

# Function to update a single HTML file
update_html_file <- function(file_path, new_url) {
  cat("Processing:", basename(file_path), "\n")
  
  # Read the HTML file
  if (!file.exists(file_path)) {
    cat("  Warning: File not found -", file_path, "\n")
    return(FALSE)
  }
  
  content <- readLines(file_path, warn = FALSE, encoding = "UTF-8")
  original_content <- content
  
  # Show current links for debugging
  show_current_links(file_path)
  
  # Pattern to match the paper link section
  updated <- FALSE
  
  for (i in seq_along(content)) {
    line <- content[i]
    
    # Pattern 1: Any Google Scholar links
    if (str_detect(line, "scholar\\.google\\.com")) {
      cat("  Found Scholar link on line", i, ":", str_trim(line), "\n")
      # Replace the entire href attribute
      if (str_detect(line, 'href="[^"]*scholar\\.google\\.com[^"]*"')) {
        content[i] <- str_replace(line, 
                                 'href="[^"]*scholar\\.google\\.com[^"]*"',
                                 paste0('href="', new_url, '"'))
        updated <- TRUE
        cat("  Updated href attribute\n")
      } else {
        # Handle cases where the URL might not be in href
        content[i] <- str_replace_all(line, 
                                     "https?://scholar\\.google\\.com[^\\s\"'<>]*",
                                     new_url)
        updated <- TRUE
        cat("  Updated embedded Scholar URL\n")
      }
    }
    
    # Pattern 2: Update link text
    if (str_detect(line, "Search for Original Paper|Google Scholar Search|Find Original Paper")) {
      content[i] <- str_replace_all(line, 
                                   "Search for Original Paper|Google Scholar Search|Find Original Paper",
                                   "View Original Paper")
      updated <- TRUE
      cat("  Updated link text on line", i, "\n")
    }
    
    # Pattern 3: Look for any citation/paper links that might need updating
    if (str_detect(line, "paper-link|citation.*href|Find.*Paper")) {
      if (!str_detect(line, new_url) && !str_detect(line, "scholar\\.google\\.com")) {
        cat("  Found potential paper link that might need manual review on line", i, ":\n")
        cat("    ", str_trim(line), "\n")
      }
    }
  }
  
  if (updated) {
    # Create backup
    backup_path <- paste0(file_path, ".backup")
    file.copy(file_path, backup_path)
    cat("  Created backup:", backup_path, "\n")
    
    # Write updated content
    writeLines(content, file_path, useBytes = TRUE)
    cat("  Successfully updated:", basename(file_path), "\n")
    return(TRUE)
  } else {
    cat("  No Google Scholar links found to update\n")
    
    # Check if file needs manual review
    content_string <- paste(content, collapse = " ")
    if (str_detect(content_string, "paper.*link|citation|original.*paper") && 
        !str_detect(content_string, fixed(new_url))) {
      cat("  NOTE: File may need manual review - contains paper references but no Scholar links\n")
    }
    
    return(FALSE)
  }
}

# Function to process all HTML files in a directory
process_all_files <- function(reviews_dir = "reviews") {
  if (!dir.exists(reviews_dir)) {
    reviews_dir <- "."  # Use current directory if reviews folder doesn't exist
    cat("Reviews directory not found, using current directory\n")
  }
  
  cat("Starting batch update of HTML review files...\n")
  cat("Working directory:", getwd(), "\n")
  cat("Reviews directory:", reviews_dir, "\n\n")
  
  updated_count <- 0
  not_found_count <- 0
  no_updates_count <- 0
  
  for (filename in names(paper_links)) {
    file_path <- file.path(reviews_dir, filename)
    new_url <- paper_links[[filename]]
    
    cat("\n--- Processing", filename, "---\n")
    cat("Target URL:", new_url, "\n")
    
    if (file.exists(file_path)) {
      if (update_html_file(file_path, new_url)) {
        updated_count <- updated_count + 1
      } else {
        no_updates_count <- no_updates_count + 1
      }
    } else {
      cat("File not found:", file_path, "\n")
      not_found_count <- not_found_count + 1
    }
  }
  
  cat("\n", paste(rep("=", 50), collapse = ""), "\n")
  cat("SUMMARY:\n")
  cat("Files updated:", updated_count, "\n")
  cat("Files with no updates needed:", no_updates_count, "\n")
  cat("Files not found:", not_found_count, "\n")
  cat("Total files processed:", length(paper_links), "\n")
  
  if (updated_count > 0) {
    cat("\nBackup files created with .backup extension\n")
    cat("Review the changes and delete backups when satisfied\n")
  }
}

# Function to verify updates
verify_updates <- function(reviews_dir = "reviews") {
  cat("Verifying updates...\n\n")
  
  for (filename in names(paper_links)) {
    file_path <- file.path(reviews_dir, filename)
    expected_url <- paper_links[[filename]]
    
    if (file.exists(file_path)) {
      content <- readLines(file_path, warn = FALSE)
      content_string <- paste(content, collapse = " ")
      
      has_scholar <- str_detect(content_string, "scholar\\.google\\.com")
      has_expected <- str_detect(content_string, fixed(expected_url))
      
      cat(filename, ":\n")
      cat("  Has Google Scholar links:", has_scholar, "\n")
      cat("  Has expected URL:", has_expected, "\n")
      
      if (has_scholar) {
        cat("  WARNING: Still contains Google Scholar links!\n")
      }
      if (!has_expected && !has_scholar) {
        cat("  INFO: No expected URL found (might not have had links to update)\n")
      }
      cat("\n")
    }
  }
}

# Function to clean up backup files
cleanup_backups <- function(reviews_dir = "reviews") {
  backup_files <- list.files(reviews_dir, pattern = "\\.backup$", full.names = TRUE)
  
  if (length(backup_files) > 0) {
    cat("Found", length(backup_files), "backup files:\n")
    for (bf in backup_files) {
      cat(" ", basename(bf), "\n")
    }
    
    response <- readline(prompt = "Delete all backup files? (y/N): ")
    if (tolower(response) == "y") {
      file.remove(backup_files)
      cat("Backup files deleted.\n")
    } else {
      cat("Backup files preserved.\n")
    }
  } else {
    cat("No backup files found.\n")
  }
}

# Main execution
main <- function() {
  cat("HTML Review Files Paper Link Updater\n")
  cat("====================================\n\n")
  
  # Check if we're in the right directory
  if (file.exists("index.html")) {
    cat("Found index.html - appears to be in correct directory\n")
  } else {
    cat("Warning: index.html not found - you may need to adjust the working directory\n")
  }
  
  # Process all files
  process_all_files()
  
  # Verify the updates
  cat("\n")
  verify_updates()
  
  # Option to clean up backups
  cat("\n")
  cleanup_backups()
}

# Run the main function
main()

# Function to manually inspect a specific file
inspect_file <- function(filename, reviews_dir = "reviews") {
  file_path <- file.path(reviews_dir, filename)
  
  if (!file.exists(file_path)) {
    cat("File not found:", file_path, "\n")
    return()
  }
  
  cat("Inspecting:", filename, "\n")
  cat("=" , rep("=", nchar(filename) + 10), "\n")
  
  content <- readLines(file_path, warn = FALSE, encoding = "UTF-8")
  
  # Show all links
  cat("\nAll href links:\n")
  for (i in seq_along(content)) {
    if (str_detect(content[i], 'href="')) {
      matches <- str_extract_all(content[i], 'href="[^"]*"')[[1]]
      if (length(matches) > 0) {
        cat("Line", i, ":")
        for (match in matches) {
          cat(" ", match)
        }
        cat("\n")
      }
    }
  }
  
  # Show citation/paper related lines
  cat("\nLines with 'paper', 'citation', or 'original':\n")
  for (i in seq_along(content)) {
    if (str_detect(tolower(content[i]), "paper|citation|original")) {
      cat("Line", i, ":", str_trim(content[i]), "\n")
    }
  }
  
  # Check for Google Scholar
  scholar_found <- any(str_detect(content, "scholar\\.google\\.com"))
  cat("\nContains Google Scholar links:", scholar_found, "\n")
  
  if (scholar_found) {
    cat("Scholar lines:\n")
    for (i in seq_along(content)) {
      if (str_detect(content[i], "scholar\\.google\\.com")) {
        cat("Line", i, ":", str_trim(content[i]), "\n")
      }
    }
  }
}
add_paper_link <- function(filename, url) {
  cat("Adding new paper link mapping:\n")
  cat("File:", filename, "\n")
  cat("URL:", url, "\n")
  cat("Add this to the paper_links list in the script:\n")
  cat(paste0('"', filename, '" = "', url, '",\n'))
}

# Function to check for any remaining Google Scholar links across all HTML files
check_remaining_scholar_links <- function(reviews_dir = "reviews") {
  cat("Checking for any remaining Google Scholar links...\n\n")
  
  html_files <- list.files(reviews_dir, pattern = "\\.html$", full.names = TRUE)
  
  for (file_path in html_files) {
    content <- readLines(file_path, warn = FALSE)
    content_string <- paste(content, collapse = " ")
    
    if (str_detect(content_string, "scholar\\.google\\.com")) {
      cat("Found Google Scholar links in:", basename(file_path), "\n")
      
      # Extract the lines with scholar links
      scholar_lines <- content[str_detect(content, "scholar\\.google\\.com")]
      for (line in scholar_lines) {
        cat("  ", str_trim(line), "\n")
      }
      cat("\n")
    }
  }
}

cat("Script loaded successfully!\n")
cat("Run main() to start the update process\n")
cat("Available functions:\n")
cat("  main() - Run the complete update process\n")
cat("  process_all_files() - Update all files\n")
cat("  verify_updates() - Check if updates were successful\n")
cat("  cleanup_backups() - Remove backup files\n")
cat("  check_remaining_scholar_links() - Find any remaining Scholar links\n")