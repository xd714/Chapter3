# R Script to Replace Theme Algorithm in HTML Files
# This script finds and replaces the old theme algorithm with the improved version

library(stringr)

# Define the new improved algorithm
new_algorithm <- '
    function initTheme() {
        const filename = location.pathname.split(\'/\').pop().replace(\'.html\', \'\') || \'default\';
        
        // Ultra-simple but effective distribution
        let seed = 0;
        
        // Use first and last characters heavily
        if (filename.length > 0) {
            seed += filename.charCodeAt(0) * 13;
            seed += filename.charCodeAt(filename.length - 1) * 17;
        }
        
        // Add middle character if exists
        if (filename.length > 2) {
            const mid = Math.floor(filename.length / 2);
            seed += filename.charCodeAt(mid) * 19;
        }
        
        // Add length and vowel count
        seed += filename.length * 23;
        const vowelCount = (filename.match(/[aeiouAEIOU]/g) || []).length;
        seed += vowelCount * 29;
        
        // Extract year and add to seed
        const year = filename.match(/\\d{4}/)?.[0];
        if (year) {
            seed += parseInt(year) * 7;
        }
        
        // Simple modulo for theme number
        const themeNumber = (seed % 30) + 1;
        
        document.body.setAttribute(\'data-theme\', \'auto-\' + themeNumber);
        if (year) document.body.setAttribute(\'data-year\', year);
    }'

# Function to replace theme algorithm in a single file
replace_theme_in_file <- function(file_path) {
  tryCatch({
    # Read the file
    content <- readLines(file_path, warn = FALSE, encoding = "UTF-8")
    content_text <- paste(content, collapse = "\n")
    
    # Check if file contains the old algorithm pattern
    if (!str_detect(content_text, "function initTheme\\(\\)")) {
      cat("No initTheme function found in:", file_path, "\n")
      return(FALSE)
    }
    
    # Pattern to match the old initTheme function
    # This matches from "function initTheme()" to the closing brace
    old_pattern <- "function initTheme\\(\\)\\s*\\{[^}]*(?:\\{[^}]*\\}[^}]*)*\\}"
    
    # Replace the old function with the new one
    new_content <- str_replace(content_text, old_pattern, new_algorithm)
    
    # Check if replacement was made
    if (identical(content_text, new_content)) {
      cat("No replacement made in:", file_path, "(pattern not matched)\n")
      return(FALSE)
    }
    
    # Write back to file
    writeLines(strsplit(new_content, "\n")[[1]], file_path, useBytes = TRUE)
    
    cat("✓ Successfully updated:", file_path, "\n")
    return(TRUE)
    
  }, error = function(e) {
    cat("✗ Error processing", file_path, ":", e$message, "\n")
    return(FALSE)
  })
}

# Function to process all HTML files in a directory
update_html_files <- function(directory_path = ".", recursive = TRUE, backup = TRUE) {
  # Validate directory
  if (!dir.exists(directory_path)) {
    stop("Directory does not exist: ", directory_path)
  }
  
  # Find all HTML files
  html_files <- list.files(
    path = directory_path,
    pattern = "\\.html?$",
    full.names = TRUE,
    recursive = recursive,
    ignore.case = TRUE
  )
  
  if (length(html_files) == 0) {
    cat("No HTML files found in:", directory_path, "\n")
    return(invisible(NULL))
  }
  
  cat("Found", length(html_files), "HTML files\n")
  
  # Create backup directory if requested
  if (backup) {
    backup_dir <- file.path(directory_path, "backup_themes")
    if (!dir.exists(backup_dir)) {
      dir.create(backup_dir, recursive = TRUE)
      cat("Created backup directory:", backup_dir, "\n")
    }
  }
  
  # Process each file
  results <- data.frame(
    file = character(0),
    status = character(0),
    stringsAsFactors = FALSE
  )
  
  for (file_path in html_files) {
    # Create backup if requested
    if (backup) {
      backup_path <- file.path(backup_dir, basename(file_path))
      file.copy(file_path, backup_path, overwrite = TRUE)
    }
    
    # Process the file
    success <- replace_theme_in_file(file_path)
    
    results <- rbind(results, data.frame(
      file = basename(file_path),
      status = ifelse(success, "Updated", "No change"),
      stringsAsFactors = FALSE
    ))
  }
  
  # Summary
  cat("\n=== SUMMARY ===\n")
  cat("Total files processed:", nrow(results), "\n")
  cat("Successfully updated:", sum(results$status == "Updated"), "\n")
  cat("No changes needed:", sum(results$status == "No change"), "\n")
  
  if (backup && sum(results$status == "Updated") > 0) {
    cat("Backups saved in:", backup_dir, "\n")
  }
  
  # Return results
  invisible(results)
}

# Function to test the algorithm on a sample filename
test_new_algorithm <- function(filename) {
  # Simulate the JavaScript algorithm in R
  filename_clean <- gsub("\\.html$", "", filename)
  
  # Ultra-simple but effective distribution
  seed <- 0
  
  # Use first and last characters heavily
  if (nchar(filename_clean) > 0) {
    seed <- seed + utf8ToInt(substr(filename_clean, 1, 1)) * 13
    seed <- seed + utf8ToInt(substr(filename_clean, nchar(filename_clean), nchar(filename_clean))) * 17
  }
  
  # Add middle character if exists
  if (nchar(filename_clean) > 2) {
    mid <- floor(nchar(filename_clean) / 2)
    seed <- seed + utf8ToInt(substr(filename_clean, mid, mid)) * 19
  }
  
  # Add length and vowel count
  seed <- seed + nchar(filename_clean) * 23
  vowel_count <- length(gregexpr("[aeiouAEIOU]", filename_clean)[[1]])
  if (vowel_count > 0 && gregexpr("[aeiouAEIOU]", filename_clean)[[1]][1] != -1) {
    seed <- seed + vowel_count * 29
  }
  
  # Extract year and add to seed
  year_match <- regmatches(filename_clean, regexpr("\\d{4}", filename_clean))
  if (length(year_match) > 0) {
    seed <- seed + as.numeric(year_match[1]) * 7
  }
  
  # Simple modulo for theme number
  theme_number <- (seed %% 30) + 1
  
  cat("Filename:", filename, "-> Theme:", theme_number, "\n")
  return(theme_number)
}

# USAGE EXAMPLES:

# 1. Update all HTML files in current directory (with backup)
# results <- update_html_files()

# 2. Update files in specific directory without backup
# results <- update_html_files("path/to/your/html/files", backup = FALSE)

# 3. Update files non-recursively (only current directory)
# results <- update_html_files(recursive = FALSE)

# 4. Test the new algorithm on sample filenames
cat("=== TESTING NEW ALGORITHM ===\n")
test_files <- c(
  "literature_review_2023.html",
  "analysis_genomics_2024.html", 
  "study_cancer_research.html",
  "paper_machine_learning.html",
  "review_ai_methods_2022.html"
)

for (file in test_files) {
  test_new_algorithm(file)
}

cat("\n=== READY TO RUN ===\n")
cat("Uncomment one of the usage examples above to start processing files.\n")
cat("Example: results <- update_html_files()\n")

# Optional: Interactive mode
run_interactive <- function() {
  cat("=== INTERACTIVE MODE ===\n")
  dir_path <- readline("Enter directory path (or press Enter for current directory): ")
  if (dir_path == "") dir_path <- "."
  
  backup_choice <- readline("Create backups? (y/n): ")
  backup <- tolower(backup_choice) %in% c("y", "yes", "1", "true")
  
  recursive_choice <- readline("Process subdirectories recursively? (y/n): ")
  recursive <- tolower(recursive_choice) %in% c("y", "yes", "1", "true")
  
  cat("Processing files...\n")
  results <- update_html_files(dir_path, recursive = recursive, backup = backup)
  
  return(results)
}

# Uncomment to run in interactive mode:
# results <- run_interactive()