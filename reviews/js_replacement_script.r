# R Script to Replace JavaScript in Multiple HTML Files
# This script will find all HTML files and replace the long JavaScript with a shorter version

library(stringr)

# Define the new shortened JavaScript
new_js <- '
<script>
(function() {
    \'use strict\';
    
    // Auto-theme based on filename
    function initTheme() {
        const filename = location.pathname.split(\'/\').pop().replace(\'.html\', \'\') || \'default\';
        let hash = 5381;
        for (let i = 0; i < filename.length; i++) {
            hash = ((hash << 5) + hash) + filename.charCodeAt(i);
        }
        const themeNumber = (Math.abs(hash) % 20) + 1;
        document.body.setAttribute(\'data-theme\', \'auto-\' + themeNumber);
        
        // Extract year if present
        const year = filename.match(/\\d{4}/)?.[0];
        if (year) document.body.setAttribute(\'data-year\', year);
    }
    
    // Back to top button
    function initBackToTop() {
        const btn = document.createElement(\'div\');
        btn.className = \'back-to-top\';
        btn.setAttribute(\'aria-label\', \'Back to top\');
        btn.onclick = () => window.scrollTo({top: 0, behavior: \'smooth\'});
        document.body.appendChild(btn);
        
        window.onscroll = () => {
            btn.classList.toggle(\'visible\', window.pageYOffset > 300);
        };
    }
    
    // Initialize when ready
    document.readyState === \'loading\' 
        ? document.addEventListener(\'DOMContentLoaded\', () => {initTheme(); initBackToTop();})
        : (initTheme(), initBackToTop());
})();
</script>'

# Function to process a single HTML file
process_html_file <- function(file_path) {
  cat("Processing:", file_path, "\n")
  
  # Read the file
  content <- readLines(file_path, warn = FALSE, encoding = "UTF-8")
  content_text <- paste(content, collapse = "\n")
  
  # Find the start and end of the script tag
  script_start <- str_locate(content_text, "<script>")[1]
  script_end <- str_locate(content_text, "</script>")[2]
  
  if (is.na(script_start) || is.na(script_end)) {
    cat("  Warning: No <script> tags found in", file_path, "\n")
    return(FALSE)
  }
  
  # Replace the JavaScript section
  before_script <- str_sub(content_text, 1, script_start - 1)
  after_script <- str_sub(content_text, script_end + 1, -1)
  
  new_content <- paste0(before_script, new_js, after_script)
  
  # Create backup
  backup_path <- paste0(file_path, ".backup")
  file.copy(file_path, backup_path, overwrite = TRUE)
  cat("  Backup created:", backup_path, "\n")
  
  # Write the new content
  writeLines(new_content, file_path, useBytes = TRUE)
  cat("  ✓ JavaScript replaced successfully\n")
  
  return(TRUE)
}

# Main execution
main <- function() {
  cat("=== HTML JavaScript Replacement Script ===\n\n")
  
  # Get current working directory
  current_dir <- getwd()
  cat("Working directory:", current_dir, "\n\n")
  
  # Find all HTML files in current directory
  html_files <- list.files(pattern = "\\.html$", full.names = TRUE)
  
  if (length(html_files) == 0) {
    cat("No HTML files found in current directory.\n")
    cat("Make sure you're in the correct folder with your HTML files.\n")
    return()
  }
  
  cat("Found", length(html_files), "HTML files:\n")
  for (file in html_files) {
    cat(" -", basename(file), "\n")
  }
  cat("\n")
  
  # Ask for confirmation
  cat("This will:\n")
  cat("1. Create .backup files for all HTML files\n")
  cat("2. Replace the JavaScript section in each file\n")
  cat("3. Preserve all other content\n\n")
  
  response <- readline(prompt = "Do you want to proceed? (y/n): ")
  
  if (tolower(response) != "y" && tolower(response) != "yes") {
    cat("Operation cancelled.\n")
    return()
  }
  
  # Process each file
  cat("\nProcessing files...\n")
  success_count <- 0
  
  for (file in html_files) {
    if (process_html_file(file)) {
      success_count <- success_count + 1
    }
    cat("\n")
  }
  
  # Summary
  cat("=== Summary ===\n")
  cat("Files processed successfully:", success_count, "/", length(html_files), "\n")
  cat("Backup files created with .backup extension\n")
  
  if (success_count == length(html_files)) {
    cat("✓ All files processed successfully!\n")
  } else {
    cat("⚠ Some files had issues. Check the output above.\n")
  }
  
  cat("\nTo undo changes, you can restore from .backup files:\n")
  cat("Example: file.copy('filename.html.backup', 'filename.html', overwrite = TRUE)\n")
}

# Run the script
main()

# Additional utility functions for manual control
restore_from_backup <- function(filename) {
  backup_file <- paste0(filename, ".backup")
  if (file.exists(backup_file)) {
    file.copy(backup_file, filename, overwrite = TRUE)
    cat("Restored", filename, "from backup\n")
  } else {
    cat("Backup file not found:", backup_file, "\n")
  }
}

# Function to restore all files from backup
restore_all_from_backup <- function() {
  backup_files <- list.files(pattern = "\\.html\\.backup$")
  for (backup in backup_files) {
    original <- str_remove(backup, "\\.backup$")
    file.copy(backup, original, overwrite = TRUE)
    cat("Restored:", original, "\n")
  }
  cat("All files restored from backup\n")
}

# Clean up backup files (run this after you're satisfied with the changes)
clean_backups <- function() {
  backup_files <- list.files(pattern = "\\.html\\.backup$")
  file.remove(backup_files)
  cat("Removed", length(backup_files), "backup files\n")
}