# HTML Theme Updater Script
# This script automatically updates all HTML files with the improved theme system

# Load required libraries
library(stringr)

# Define the new JavaScript code
new_script <- '
<script>
(function() {
    \'use strict\';
    
    // Universal auto-theme with maximum color variety
    function initTheme() {
        const filename = location.pathname.split(\'/\').pop().replace(\'.html\', \'\') || \'default\';
        const themeNumber = getPerfectTheme(filename);
        document.body.setAttribute(\'data-theme\', \'auto-\' + themeNumber);
        
        // Extract year if present
        const year = filename.match(/\\d{4}/)?.[0];
        if (year) document.body.setAttribute(\'data-year\', year);
    }
    
    function getPerfectTheme(filename) {
        // Prioritize visually distinct themes, minimize greens
        const visuallyDistinctThemes = [
            1,  // Red
            2,  // Blue  
            4,  // Orange
            5,  // Purple
            11, // Yellow
            12, // Pink
            13, // Cyan
            14, // Deep Orange
            9,  // Dark Purple
            7,  // Dark Orange
            16, // Brown
            6,  // Teal
            17, // Amber
            15, // Blue Gray
            10, // Dark Blue-Gray
            19, // Deep Purple
            20, // Teal Green
            3,  // Green (lower priority)
            8,  // Light Green (lower priority)
            18  // Light Green (lowest priority)
        ];
        
        // Multi-source hash for maximum distribution
        let primaryHash = 0;
        let secondaryHash = 0;
        
        // 1. Character processing with position weighting
        for (let i = 0; i < filename.length; i++) {
            const char = filename.charCodeAt(i);
            primaryHash += char * (i + 1) * 37; // Prime multiplier
            secondaryHash ^= char << (i % 16);   // Bit shifting
        }
        
        // 2. Year influence (if present)
        const year = filename.match(/\\d{4}/)?.[0];
        if (year) {
            primaryHash += parseInt(year) * 4993; // Large prime
            secondaryHash += (parseInt(year) % 100) * 73;
        }
        
        // 3. Pattern analysis for extra differentiation
        const vowels = (filename.match(/[aeiou]/gi) || []).length;
        const consonants = (filename.match(/[bcdfghjklmnpqrstvwxyz]/gi) || []).length;
        const numbers = (filename.match(/\\d/g) || []).length;
        
        primaryHash += vowels * 127 + consonants * 149 + numbers * 163;
        
        // 4. Final theme selection
        const combinedHash = Math.abs(primaryHash ^ secondaryHash);
        const themeIndex = combinedHash % visuallyDistinctThemes.length;
        
        return visuallyDistinctThemes[themeIndex];
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

# Function to update HTML files
update_html_files <- function(directory = ".", backup = TRUE) {
  
  cat("🔍 Searching for HTML files...\n")
  
  # Find all HTML files in the directory
  html_files <- list.files(directory, pattern = "\\.html$", full.names = TRUE, recursive = FALSE)
  
  if (length(html_files) == 0) {
    cat("❌ No HTML files found in", directory, "\n")
    return(FALSE)
  }
  
  cat("📁 Found", length(html_files), "HTML files:\n")
  for (file in html_files) {
    cat("   -", basename(file), "\n")
  }
  
  # Create backup directory if needed
  if (backup) {
    backup_dir <- file.path(directory, "backup_original_scripts")
    if (!dir.exists(backup_dir)) {
      dir.create(backup_dir)
      cat("📦 Created backup directory:", backup_dir, "\n")
    }
  }
  
  # Process each HTML file
  updated_count <- 0
  skipped_count <- 0
  
  for (html_file in html_files) {
    cat("\n🔧 Processing:", basename(html_file), "\n")
    
    # Read the file
    tryCatch({
      content <- readLines(html_file, warn = FALSE)
      original_content <- paste(content, collapse = "\n")
      
      # Create backup if requested
      if (backup) {
        backup_file <- file.path(backup_dir, basename(html_file))
        writeLines(content, backup_file)
        cat("   💾 Backup saved to:", basename(backup_file), "\n")
      }
      
      # Find the script section to replace
      script_start <- which(str_detect(content, "<script>"))
      script_end <- which(str_detect(content, "</script>"))
      
      if (length(script_start) > 0 && length(script_end) > 0) {
        # Find the main script block (usually the first or largest one)
        main_script_idx <- 1
        if (length(script_start) > 1) {
          # Find the script block that contains "initTheme" or is the largest
          for (i in 1:length(script_start)) {
            script_content <- paste(content[script_start[i]:script_end[i]], collapse = "\n")
            if (str_detect(script_content, "initTheme|auto-theme")) {
              main_script_idx <- i
              break
            }
          }
        }
        
        start_line <- script_start[main_script_idx]
        end_line <- script_end[main_script_idx]
        
        cat("   🎯 Found script block at lines", start_line, "to", end_line, "\n")
        
        # Replace the script section
        new_content <- c(
          content[1:(start_line-1)],
          str_split(new_script, "\n")[[1]],
          content[(end_line+1):length(content)]
        )
        
        # Write the updated content
        writeLines(new_content, html_file)
        
        cat("   ✅ Successfully updated!\n")
        updated_count <- updated_count + 1
        
      } else {
        cat("   ⚠️  No script block found - skipping\n")
        skipped_count <- skipped_count + 1
      }
      
    }, error = function(e) {
      cat("   ❌ Error processing file:", e$message, "\n")
      skipped_count <- skipped_count + 1
    })
  }
  
  # Summary
  cat("\n" , rep("=", 50), "\n")
  cat("📊 SUMMARY:\n")
  cat("   ✅ Files updated:", updated_count, "\n")
  cat("   ⚠️  Files skipped:", skipped_count, "\n")
  cat("   💾 Backups created:", ifelse(backup, "Yes", "No"), "\n")
  
  if (updated_count > 0) {
    cat("\n🎨 Your HTML files now have improved color themes!\n")
    cat("🔍 Open any HTML file to see the new color scheme.\n")
    
    if (backup) {
      cat("🛡️  Original files backed up in: backup_original_scripts/\n")
    }
  }
  
  return(updated_count > 0)
}

# Function to preview what themes your files will get
preview_themes <- function(directory = ".") {
  html_files <- list.files(directory, pattern = "\\.html$", full.names = FALSE)
  
  if (length(html_files) == 0) {
    cat("No HTML files found.\n")
    return()
  }
  
  # Simulate the JavaScript theme algorithm in R
  get_perfect_theme <- function(filename) {
    # Remove .html extension
    filename <- str_replace(filename, "\\.html$", "")
    
    # Theme priority list (matching JavaScript)
    themes <- c(1, 2, 4, 5, 11, 12, 13, 14, 9, 7, 16, 6, 17, 15, 10, 19, 20, 3, 8, 18)
    
    # Simple hash simulation (approximate)
    primary_hash <- 0
    secondary_hash <- 0
    
    # Character processing
    chars <- utf8ToInt(filename)
    for (i in seq_along(chars)) {
      primary_hash <- primary_hash + chars[i] * i * 37
      secondary_hash <- bitwXor(secondary_hash, bitwShiftL(chars[i], (i-1) %% 16))
    }
    
    # Year processing
    year_match <- str_extract(filename, "\\d{4}")
    if (!is.na(year_match)) {
      year <- as.numeric(year_match)
      primary_hash <- primary_hash + year * 4993
      secondary_hash <- secondary_hash + (year %% 100) * 73
    }
    
    # Pattern analysis
    vowels <- str_count(filename, "[aeiouAEIOU]")
    consonants <- str_count(filename, "[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]")
    numbers <- str_count(filename, "\\d")
    
    primary_hash <- primary_hash + vowels * 127 + consonants * 149 + numbers * 163
    
    # Final selection
    combined_hash <- abs(bitwXor(primary_hash, secondary_hash))
    theme_index <- (combined_hash %% length(themes)) + 1
    
    return(themes[theme_index])
  }
  
  # Color mapping
  colors <- c(
    "1" = "Red", "2" = "Blue", "3" = "Green", "4" = "Orange", "5" = "Purple",
    "6" = "Teal", "7" = "Dark Orange", "8" = "Light Green", "9" = "Dark Purple", 
    "10" = "Dark Blue-Gray", "11" = "Yellow", "12" = "Pink", "13" = "Cyan", 
    "14" = "Deep Orange", "15" = "Blue Gray", "16" = "Brown", "17" = "Amber", 
    "18" = "Light Green Alt", "19" = "Deep Purple", "20" = "Teal Green"
  )
  
  cat("🎨 THEME PREVIEW:\n")
  cat(rep("=", 60), "\n")
  
  results <- data.frame(
    File = character(),
    Theme = numeric(),
    Color = character(),
    stringsAsFactors = FALSE
  )
  
  for (file in html_files) {
    theme <- get_perfect_theme(file)
    color <- colors[as.character(theme)]
    results <- rbind(results, data.frame(File = file, Theme = theme, Color = color))
    cat(sprintf("%-25s -> Theme %-2d = %s\n", file, theme, color))
  }
  
  cat(rep("=", 60), "\n")
  
  # Analysis
  green_themes <- c(3, 8, 18)
  green_count <- sum(results$Theme %in% green_themes)
  unique_themes <- length(unique(results$Theme))
  
  cat("📊 ANALYSIS:\n")
  cat("   Total files:", nrow(results), "\n")
  cat("   Unique themes:", unique_themes, "\n")
  cat("   Green themes:", green_count, "/", nrow(results), "\n")
  cat("   Duplicate themes:", nrow(results) - unique_themes, "\n")
  
  return(results)
}

# MAIN EXECUTION
cat("🎨 HTML Theme Updater Script\n")
cat(rep("=", 50), "\n")

# Check if we should run the update
cat("This script will:\n")
cat("1. 🔍 Find all HTML files in current directory\n")
cat("2. 💾 Create backups of original files\n")
cat("3. 🔧 Replace theme scripts with improved version\n")
cat("4. 🎨 Give you better color variety\n\n")

# Preview themes first
cat("📋 STEP 1: Preview new themes\n")
preview_results <- preview_themes()

cat("\n📝 STEP 2: Ready to update files\n")
cat("Continue with update? (y/n): ")

# For automatic execution, uncomment the next line and comment the interactive part
# user_input <- "y"

# Interactive prompt (comment out for automatic execution)
user_input <- readline()

if (tolower(trimws(user_input)) %in% c("y", "yes", "1")) {
  cat("\n🚀 Starting update process...\n")
  success <- update_html_files(backup = TRUE)
  
  if (success) {
    cat("\n🎉 All done! Your HTML files now have improved themes.\n")
    cat("🌐 Open any HTML file in your browser to see the changes.\n")
  }
} else {
  cat("\n❌ Update cancelled. No files were modified.\n")
}

cat("\n📚 Script completed.\n")