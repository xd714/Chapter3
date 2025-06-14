# Simple HTML Script Replacer - Updates ONLY the JavaScript in HTML files
# Assumes CSS is already updated

library(stringr)

# The ultra-simple JavaScript replacement
new_simple_script <- '
<script>
(function() {
    \'use strict\';
    
    function initTheme() {
        const filename = location.pathname.split(\'/\').pop().replace(\'.html\', \'\') || \'default\';
        
        // Ultra-simple hash: character codes + position weights + year
        let hash = 0;
        for (let i = 0; i < filename.length; i++) {
            hash += filename.charCodeAt(i) * (i + 1);
        }
        
        const year = filename.match(/\\d{4}/)?.[0];
        if (year) hash += parseInt(year);
        
        // 30 themes = maximum variety, minimal green clustering
        const themeNumber = (hash % 30) + 1;
        
        document.body.setAttribute(\'data-theme\', \'auto-\' + themeNumber);
        if (year) document.body.setAttribute(\'data-year\', year);
    }
    
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
    
    document.readyState === \'loading\' 
        ? document.addEventListener(\'DOMContentLoaded\', () => {initTheme(); initBackToTop();})
        : (initTheme(), initBackToTop());
})();
</script>'

# Quick preview function
preview_new_themes <- function(directory = ".") {
  html_files <- list.files(directory, pattern = "\\.html$", full.names = FALSE)
  
  if (length(html_files) == 0) {
    cat("❌ No HTML files found.\n")
    return()
  }
  
  # Simulate the ultra-simple algorithm
  get_simple_theme <- function(filename) {
    filename <- str_replace(filename, "\\.html$", "")
    
    hash <- 0
    chars <- utf8ToInt(filename)
    for (i in seq_along(chars)) {
      hash <- hash + chars[i] * i
    }
    
    year_match <- str_extract(filename, "\\d{4}")
    if (!is.na(year_match)) {
      hash <- hash + as.numeric(year_match)
    }
    
    return((hash %% 30) + 1)
  }
  
  # 30-theme color map
  colors <- c(
    "1" = "Red", "2" = "Blue", "3" = "Green", "4" = "Orange", "5" = "Purple",
    "6" = "Teal", "7" = "Dark Orange", "8" = "Light Green", "9" = "Dark Purple", 
    "10" = "Dark Blue-Gray", "11" = "Yellow", "12" = "Pink", "13" = "Cyan", 
    "14" = "Deep Orange", "15" = "Blue Gray", "16" = "Brown", "17" = "Amber", 
    "18" = "Light Green Alt", "19" = "Deep Purple", "20" = "Teal Green",
    "21" = "Coral Red", "22" = "Mint Teal", "23" = "Sky Blue", "24" = "Sage Green", 
    "25" = "Warm Yellow", "26" = "Plum", "27" = "Seafoam", "28" = "Gold", 
    "29" = "Lavender", "30" = "Powder Blue"
  )
  
  cat("🎨 NEW THEME PREVIEW:\n")
  cat(rep("=", 50), "\n")
  
  for (file in html_files) {
    theme <- get_simple_theme(file)
    color <- colors[as.character(theme)]
    cat(sprintf("%-25s -> Theme %-2d = %s\n", file, theme, color))
  }
  
  cat(rep("=", 50), "\n")
  
  # Quick stats
  themes <- sapply(html_files, get_simple_theme)
  green_themes <- c(3, 8, 18, 24)
  green_count <- sum(themes %in% green_themes)
  
  cat("📊 QUICK STATS:\n")
  cat("   Files found:", length(html_files), "\n")
  cat("   Green themes:", green_count, "/", length(html_files), 
      sprintf(" (%.1f%%)\n", green_count/length(html_files)*100))
  cat("   Unique themes:", length(unique(themes)), "\n")
}

# Main function to replace scripts in HTML files
replace_html_scripts <- function(directory = ".", backup = TRUE) {
  
  cat("🔍 Finding HTML files in:", directory, "\n")
  
  html_files <- list.files(directory, pattern = "\\.html$", full.names = TRUE)
  
  if (length(html_files) == 0) {
    cat("❌ No HTML files found.\n")
    return(FALSE)
  }
  
  cat("📁 Found", length(html_files), "HTML files\n")
  
  # Create backup directory
  if (backup) {
    backup_dir <- file.path(directory, "backup_scripts")
    if (!dir.exists(backup_dir)) {
      dir.create(backup_dir)
      cat("📦 Created backup directory:", backup_dir, "\n")
    }
  }
  
  updated_count <- 0
  
  for (html_file in html_files) {
    filename <- basename(html_file)
    cat("\n🔧", filename, "...")
    
    tryCatch({
      # Read file
      content <- readLines(html_file, warn = FALSE)
      
      # Backup
      if (backup) {
        backup_file <- file.path(backup_dir, filename)
        writeLines(content, backup_file)
      }
      
      # Find theme script
      script_starts <- which(str_detect(content, "<script>"))
      script_ends <- which(str_detect(content, "</script>"))
      
      theme_script_found <- FALSE
      
      for (i in seq_along(script_starts)) {
        if (i <= length(script_ends)) {
          start_line <- script_starts[i]
          end_line <- script_ends[i]
          
          script_block <- paste(content[start_line:end_line], collapse = "\n")
          
          # Check if this is the theme script
          if (str_detect(script_block, "initTheme|auto-theme|data-theme|getPerfectTheme|getGreenFreeTheme")) {
            
            # Replace this script block
            new_content <- c(
              content[1:(start_line-1)],
              str_split(new_simple_script, "\n")[[1]],
              content[(end_line+1):length(content)]
            )
            
            writeLines(new_content, html_file)
            cat(" ✅ Updated")
            theme_script_found <- TRUE
            updated_count <- updated_count + 1
            break
          }
        }
      }
      
      if (!theme_script_found) {
        cat(" ⚠️ No theme script found")
      }
      
    }, error = function(e) {
      cat(" ❌ Error:", e$message)
    })
  }
  
  cat("\n\n", rep("=", 40), "\n")
  cat("📊 REPLACEMENT SUMMARY:\n")
  cat("   ✅ Files updated:", updated_count, "/", length(html_files), "\n")
  cat("   💾 Backups created:", ifelse(backup, "Yes", "No"), "\n")
  cat("   🎨 New algorithm: Ultra-simple (5 lines)\n")
  
  if (updated_count > 0) {
    cat("\n🎉 SUCCESS! HTML scripts updated to ultra-simple version!\n")
    cat("🌐 Open any HTML file to see the new themes.\n")
  } else {
    cat("\n⚠️ No theme scripts were found to update.\n")
  }
  
  return(updated_count > 0)
}

# MAIN EXECUTION
cat("📝 HTML Script Replacer\n")
cat("   • Replaces <script> sections only\n")
cat("   • Assumes CSS already updated\n")
cat("   • Uses ultra-simple 5-line algorithm\n")
cat(rep("=", 40), "\n")

# Preview what themes files will get
cat("📋 STEP 1: Preview new themes\n")
preview_new_themes()

# Confirm and execute
cat("\n📝 STEP 2: Replace HTML scripts\n")
cat("Replace scripts in all HTML files? (y/n): ")

# For automatic execution, uncomment:
# user_input <- "y"

# Interactive prompt
user_input <- readline()

if (tolower(trimws(user_input)) %in% c("y", "yes", "1")) {
  cat("\n🚀 Replacing HTML scripts...\n")
  success <- replace_html_scripts(backup = TRUE)
  
  if (success) {
    cat("\n✨ All done! Your HTML files now use the ultra-simple algorithm.\n")
  }
} else {
  cat("\n❌ Cancelled. No files changed.\n")
}

cat("\n📚 Script replacer completed.\n")