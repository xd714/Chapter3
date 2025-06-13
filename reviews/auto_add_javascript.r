# ============================================
# R SCRIPT TO ADD AUTOMATIC THEME JAVASCRIPT
# TO ALL HTML FILES IN FOLDER
# ============================================

# Load required libraries
library(stringr)
library(fs)

# JavaScript code to insert
javascript_code <- '
<script>
(function() {
    \'use strict\';
    
    function initAutoTheme() {
        try {
            const filename = window.location.pathname.split(\'/\').pop().replace(\'.html\', \'\') || \'default\';
            const pageTitle = document.title || \'\';
            
            function generateHash(str) {
                let hash = 5381;
                for (let i = 0; i < str.length; i++) {
                    hash = ((hash << 5) + hash) + str.charCodeAt(i);
                    hash = hash & hash;
                }
                return Math.abs(hash);
            }
            
            const hashValue = generateHash(filename + pageTitle);
            const themeNumber = (hashValue % 20) + 1;
            
            document.body.setAttribute(\'data-theme\', \'auto-\' + themeNumber);
            document.body.setAttribute(\'data-file\', filename);
            
            const year = filename.match(/\\d{4}/)?.[0] || pageTitle.match(/\\d{4}/)?.[0] || \'\';
            if (year) {
                document.body.setAttribute(\'data-year\', year);
            }
            
            console.log(\'🎨 Theme:\', \'auto-\' + themeNumber, \'| File:\', filename);
            
        } catch (error) {
            console.warn(\'Theme initialization failed:\', error);
            document.body.setAttribute(\'data-theme\', \'auto-1\');
        }
    }
    
    function initBackToTop() {
        try {
            let backToTopButton = document.getElementById(\'backToTop\') || 
                                 document.querySelector(\'.back-to-top\');
            
            if (!backToTopButton) {
                backToTopButton = document.createElement(\'div\');
                backToTopButton.className = \'back-to-top\';
                backToTopButton.id = \'backToTop\';
                backToTopButton.setAttribute(\'aria-label\', \'Back to top\');
                backToTopButton.setAttribute(\'role\', \'button\');
                backToTopButton.setAttribute(\'tabindex\', \'0\');
                document.body.appendChild(backToTopButton);
            }
            
            function toggleVisibility() {
                if (window.pageYOffset > 300) {
                    backToTopButton.classList.add(\'visible\');
                } else {
                    backToTopButton.classList.remove(\'visible\');
                }
            }
            
            function scrollToTop() {
                window.scrollTo({
                    top: 0,
                    behavior: \'smooth\'
                });
            }
            
            window.addEventListener(\'scroll\', toggleVisibility, { passive: true });
            backToTopButton.addEventListener(\'click\', scrollToTop);
            
            backToTopButton.addEventListener(\'keydown\', function(e) {
                if (e.key === \'Enter\' || e.key === \' \') {
                    e.preventDefault();
                    scrollToTop();
                }
            });
            
            toggleVisibility();
            
        } catch (error) {
            console.warn(\'Back to top initialization failed:\', error);
        }
    }
    
    function init() {
        initAutoTheme();
        initBackToTop();
    }
    
    if (document.readyState === \'loading\') {
        document.addEventListener(\'DOMContentLoaded\', init);
    } else {
        init();
    }
})();
</script>'

# Back to top button HTML to add to body
back_to_top_html <- '\n    <!-- BACK TO TOP BUTTON -->\n    <div class="back-to-top" id="backToTop"></div>'

# Function to process a single HTML file
process_html_file <- function(file_path) {
    cat("Processing:", basename(file_path), "\n")
    
    # Read the file
    content <- readLines(file_path, warn = FALSE, encoding = "UTF-8")
    content <- paste(content, collapse = "\n")
    
    # Check if JavaScript is already added
    if (str_detect(content, "initAutoTheme")) {
        cat("  → JavaScript already exists, skipping\n")
        return(FALSE)
    }
    
    # Find </head> tag and insert JavaScript before it
    if (str_detect(content, "</head>")) {
        content <- str_replace(content, 
                              "</head>", 
                              paste0(javascript_code, "\n</head>"))
        cat("  → Added JavaScript to <head>\n")
    } else {
        cat("  → Warning: No </head> tag found\n")
        return(FALSE)
    }
    
    # Find </body> tag and insert back-to-top button before it
    if (str_detect(content, "</body>")) {
        # Check if back-to-top button already exists
        if (!str_detect(content, "back-to-top")) {
            content <- str_replace(content, 
                                  "</body>", 
                                  paste0(back_to_top_html, "\n</body>"))
            cat("  → Added back-to-top button\n")
        } else {
            cat("  → Back-to-top button already exists\n")
        }
    } else {
        cat("  → Warning: No </body> tag found\n")
    }
    
    # Write the modified content back to file
    writeLines(content, file_path, useBytes = TRUE)
    cat("  → File updated successfully\n\n")
    
    return(TRUE)
}

# Function to create backup of files
create_backup <- function(folder_path) {
    backup_folder <- file.path(folder_path, "backup_html_files")
    
    if (!dir.exists(backup_folder)) {
        dir.create(backup_folder)
        cat("Created backup folder:", backup_folder, "\n")
    }
    
    html_files <- list.files(folder_path, pattern = "\\.html$", full.names = TRUE)
    html_files <- html_files[!str_detect(html_files, "backup_html_files")]
    
    for (file in html_files) {
        backup_file <- file.path(backup_folder, basename(file))
        file.copy(file, backup_file, overwrite = TRUE)
    }
    
    cat("Backed up", length(html_files), "HTML files\n\n")
}

# Main function to process all HTML files
process_all_html_files <- function(folder_path = ".") {
    cat("============================================\n")
    cat("AUTOMATIC JAVASCRIPT INJECTION FOR HTML FILES\n")
    cat("============================================\n\n")
    
    # Validate folder
    if (!dir.exists(folder_path)) {
        stop("Folder does not exist: ", folder_path)
    }
    
    # Create backup first
    cat("Step 1: Creating backup...\n")
    create_backup(folder_path)
    
    # Find all HTML files
    html_files <- list.files(folder_path, pattern = "\\.html$", full.names = TRUE)
    html_files <- html_files[!str_detect(html_files, "backup_html_files")]
    
    if (length(html_files) == 0) {
        cat("No HTML files found in:", folder_path, "\n")
        return()
    }
    
    cat("Step 2: Processing", length(html_files), "HTML files...\n\n")
    
    # Process each file
    successful <- 0
    for (file in html_files) {
        if (process_html_file(file)) {
            successful <- successful + 1
        }
    }
    
    # Summary
    cat("============================================\n")
    cat("PROCESSING COMPLETE\n")
    cat("============================================\n")
    cat("Total files found:", length(html_files), "\n")
    cat("Successfully processed:", successful, "\n")
    cat("Skipped (already had JS):", length(html_files) - successful, "\n")
    cat("Backup location: backup_html_files/\n\n")
    
    if (successful > 0) {
        cat("✅ JavaScript successfully added to", successful, "files!\n")
        cat("🎨 Each file will now automatically get a unique theme color\n")
        cat("🔝 Back-to-top buttons added where needed\n\n")
        
        cat("Files processed:\n")
        for (file in html_files) {
            cat("  →", basename(file), "\n")
        }
    }
}

# Additional utility functions

# Function to remove JavaScript (if needed to undo)
remove_javascript <- function(folder_path = ".") {
    cat("Removing JavaScript from HTML files...\n\n")
    
    html_files <- list.files(folder_path, pattern = "\\.html$", full.names = TRUE)
    html_files <- html_files[!str_detect(html_files, "backup_html_files")]
    
    for (file in html_files) {
        cat("Processing:", basename(file), "\n")
        
        content <- readLines(file, warn = FALSE, encoding = "UTF-8")
        content <- paste(content, collapse = "\n")
        
        # Remove the JavaScript block
        content <- str_replace(content, 
                              "<script>[\\s\\S]*?initAutoTheme[\\s\\S]*?</script>\\s*", 
                              "")
        
        # Remove back-to-top button
        content <- str_replace(content, 
                              "\\s*<!-- BACK TO TOP BUTTON -->\\s*<div class=\"back-to-top\"[^>]*></div>\\s*", 
                              "")
        
        writeLines(content, file, useBytes = TRUE)
        cat("  → JavaScript removed\n")
    }
    
    cat("\n✅ JavaScript removed from all files\n")
}

# Function to check which files have JavaScript
check_javascript_status <- function(folder_path = ".") {
    html_files <- list.files(folder_path, pattern = "\\.html$", full.names = TRUE)
    html_files <- html_files[!str_detect(html_files, "backup_html_files")]
    
    cat("JavaScript Status Check:\n")
    cat("========================\n")
    
    has_js <- 0
    no_js <- 0
    
    for (file in html_files) {
        content <- paste(readLines(file, warn = FALSE), collapse = "\n")
        
        if (str_detect(content, "initAutoTheme")) {
            cat("✅", basename(file), "- Has JavaScript\n")
            has_js <- has_js + 1
        } else {
            cat("❌", basename(file), "- Missing JavaScript\n")
            no_js <- no_js + 1
        }
    }
    
    cat("\nSummary:\n")
    cat("Files with JavaScript:", has_js, "\n")
    cat("Files missing JavaScript:", no_js, "\n")
}

# ============================================
# USAGE INSTRUCTIONS
# ============================================

cat("R Script for Adding Automatic Theme JavaScript\n")
cat("===============================================\n\n")

cat("Available functions:\n")
cat("1. process_all_html_files() - Add JavaScript to all HTML files\n")
cat("2. check_javascript_status() - Check which files have JavaScript\n") 
cat("3. remove_javascript() - Remove JavaScript from all files (undo)\n\n")

cat("Examples:\n")
cat('# Process all HTML files in current directory:\n')
cat('process_all_html_files()\n\n')

cat('# Process HTML files in specific folder:\n')
cat('process_all_html_files("path/to/your/folder")\n\n')

cat('# Check status:\n')
cat('check_javascript_status()\n\n')

cat("READY TO RUN! Execute: process_all_html_files()\n")

# ============================================
# UNCOMMENT THE LINE BELOW TO RUN IMMEDIATELY
# ============================================

# process_all_html_files()  # Uncomment this line to run automatically