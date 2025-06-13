# ============================================
# HTML STYLE UPDATER R SCRIPT
# Automatically replace inline styles with CSS classes
# ============================================

# Install required packages if not already installed
if (!require(stringr)) install.packages("stringr")
if (!require(glue)) install.packages("glue")

library(stringr)
library(glue)

# ============================================
# CONFIGURATION
# ============================================

# Set your folder path (change this to your actual folder path)
folder_path <- "."  # Current directory, or use something like "C:/path/to/your/html/files"

# Define the mappings of inline styles to CSS classes
style_mappings <- list(
  # Header info box
  list(
    pattern = 'style="margin-top: 15px; padding: 10px; background: rgba\\(255,255,255,0\\.1\\); border-radius: 5px;"',
    replacement = 'class="header-info"'
  ),
  
  # Citation/info boxes with blue border
  list(
    pattern = 'style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; margin-bottom: 30px;"',
    replacement = 'class="citation-box"'
  ),
  
  # Legacy/success boxes with green border
  list(
    pattern = 'style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #27ae60; margin-bottom: 20px;"',
    replacement = 'class="legacy-box"'
  ),
  
  # Small text
  list(
    pattern = 'style="margin: 0; font-size: 0\\.9em;"',
    replacement = 'class="text-small"'
  ),
  
  # Small text with top margin
  list(
    pattern = 'style="margin: 5px 0 0 0; font-size: 0\\.9em;"',
    replacement = 'class="text-small text-top"'
  ),
  
  # Text with top margin
  list(
    pattern = 'style="margin: 10px 0 0 0;"',
    replacement = 'class="text-top-large"'
  ),
  
  # Lists with top margin
  list(
    pattern = 'style="margin-top: 10px;"',
    replacement = 'class="list-top"'
  ),
  
  # Small lists
  list(
    pattern = 'style="margin: 5px 0; font-size: 0\\.9em;"',
    replacement = 'class="list-small"'
  ),
  
  # Link styles (keep as inline for now, but could be classed)
  list(
    pattern = 'style="color: #74b9ff; text-decoration: none; font-weight: bold; border: 1px solid #74b9ff; padding: 5px 10px; border-radius: 3px; display: inline-block;"',
    replacement = 'class="paper-link"'
  )
)

# ============================================
# FUNCTIONS
# ============================================

#' Read HTML file and return content
read_html_file <- function(file_path) {
  content <- readLines(file_path, warn = FALSE, encoding = "UTF-8")
  return(paste(content, collapse = "\n"))
}

#' Write content back to HTML file
write_html_file <- function(file_path, content) {
  writeLines(content, file_path, useBytes = TRUE)
}

#' Replace inline styles with CSS classes
replace_inline_styles <- function(html_content) {
  updated_content <- html_content
  replacements_made <- 0
  
  # Apply each style mapping
  for (mapping in style_mappings) {
    # Count matches before replacement
    matches_before <- str_count(updated_content, mapping$pattern)
    
    # Perform replacement
    updated_content <- str_replace_all(updated_content, mapping$pattern, mapping$replacement)
    
    # Count matches after replacement
    matches_after <- str_count(updated_content, mapping$pattern)
    
    # Calculate replacements made
    current_replacements <- matches_before - matches_after
    replacements_made <- replacements_made + current_replacements
    
    if (current_replacements > 0) {
      cat("  → Replaced", current_replacements, "instances of:", substr(mapping$pattern, 1, 50), "...\n")
    }
  }
  
  return(list(content = updated_content, replacements = replacements_made))
}

#' Add CSS link and theme script if not present
add_css_and_script <- function(html_content, filename) {
  # Check if CSS link already exists
  has_css_link <- str_detect(html_content, 'href="style\\.css"')
  
  # Check if theme script already exists
  has_theme_script <- str_detect(html_content, "data-theme.*auto")
  
  additions_made <- 0
  
  # Add CSS link if not present
  if (!has_css_link) {
    # Find the head section and add CSS link after title
    html_content <- str_replace(
      html_content,
      "(<title>.*?</title>)",
      "\\1\n    <link rel=\"stylesheet\" href=\"style.css\">"
    )
    cat("  → Added CSS link\n")
    additions_made <- additions_made + 1
  }
  
  # Add theme script if not present
  if (!has_theme_script) {
    # Create the theme script
    theme_script <- '    <script>
    (function() {
        const filename = window.location.pathname.split(\'/\').pop().replace(\'.html\', \'\');
        const hashValue = filename.split(\'\').reduce((a,b) => (((a << 5) - a) + b.charCodeAt(0)) & 0xffffffff, 0);
        const themeNumber = (Math.abs(hashValue) % 20) + 1;
        document.body.setAttribute(\'data-theme\', \'auto-\' + themeNumber);
        document.body.setAttribute(\'data-file\', filename);
        console.log(\'🎨 Auto-theme:\', \'auto-\' + themeNumber, \'for\', filename);
    })();
    </script>'
    
    # Add script before closing head tag
    html_content <- str_replace(
      html_content,
      "</head>",
      paste0(theme_script, "\n</head>")
    )
    cat("  → Added theme script\n")
    additions_made <- additions_made + 1
  }
  
  return(list(content = html_content, additions = additions_made))
}

#' Process a single HTML file
process_html_file <- function(file_path) {
  filename <- basename(file_path)
  cat("\n📄 Processing:", filename, "\n")
  
  # Read file
  tryCatch({
    html_content <- read_html_file(file_path)
    
    # Replace inline styles
    style_result <- replace_inline_styles(html_content)
    
    # Add CSS and script
    script_result <- add_css_and_script(style_result$content, filename)
    
    # Write back if changes were made
    total_changes <- style_result$replacements + script_result$additions
    
    if (total_changes > 0) {
      write_html_file(file_path, script_result$content)
      cat("✅ Updated", filename, "- Made", total_changes, "changes\n")
    } else {
      cat("✅", filename, "- No changes needed\n")
    }
    
    return(total_changes)
    
  }, error = function(e) {
    cat("❌ Error processing", filename, ":", e$message, "\n")
    return(0)
  })
}

# ============================================
# MAIN EXECUTION
# ============================================

main <- function() {
  cat("🚀 HTML Style Updater Starting...\n")
  cat("📁 Working directory:", getwd(), "\n")
  cat("📁 Target folder:", folder_path, "\n\n")
  
  # Get list of HTML files
  html_files <- list.files(
    path = folder_path,
    pattern = "\\.html?$",
    full.names = TRUE,
    ignore.case = TRUE
  )
  
  if (length(html_files) == 0) {
    cat("❌ No HTML files found in", folder_path, "\n")
    cat("💡 Make sure you're in the right directory and have HTML files to process.\n")
    return()
  }
  
  cat("📋 Found", length(html_files), "HTML files:\n")
  for (file in basename(html_files)) {
    cat("   -", file, "\n")
  }
  
  # Ask for confirmation
  cat("\n❓ Do you want to proceed with updating these files? (y/n): ")
  response <- readline()
  
  if (tolower(substr(response, 1, 1)) != "y") {
    cat("❌ Operation cancelled by user.\n")
    return()
  }
  
  cat("\n🔄 Processing files...\n")
  
  # Process each file
  total_files_changed <- 0
  total_changes_made <- 0
  
  for (file_path in html_files) {
    changes <- process_html_file(file_path)
    if (changes > 0) {
      total_files_changed <- total_files_changed + 1
      total_changes_made <- total_changes_made + changes
    }
  }
  
  # Summary
  cat("\n" , rep("=", 50), "\n")
  cat("📊 SUMMARY\n")
  cat(rep("=", 50), "\n")
  cat("Files processed:", length(html_files), "\n")
  cat("Files changed:", total_files_changed, "\n")
  cat("Total changes made:", total_changes_made, "\n")
  
  if (total_changes_made > 0) {
    cat("\n✅ SUCCESS! Your HTML files have been updated.\n")
    cat("📝 Next steps:\n")
    cat("   1. Make sure your CSS file is named 'style.css' (not 'sytle.css')\n")
    cat("   2. Add these new CSS classes to your style.css file:\n\n")
    
    # Print CSS classes to add
    cat("/* Add these new classes to your style.css */\n")
    cat(".header-info {\n")
    cat("    margin-top: 15px;\n")
    cat("    padding: 10px;\n")
    cat("    background: rgba(255,255,255,0.1);\n")
    cat("    border-radius: 5px;\n")
    cat("}\n\n")
    
    cat(".citation-box {\n")
    cat("    background: var(--bg-light);\n")
    cat("    padding: 20px;\n")
    cat("    border-radius: 8px;\n")
    cat("    border-left: 4px solid var(--primary-color);\n")
    cat("    margin-bottom: 30px;\n")
    cat("}\n\n")
    
    cat(".legacy-box {\n")
    cat("    background: var(--bg-light);\n")
    cat("    padding: 20px;\n")
    cat("    border-radius: 8px;\n")
    cat("    border-left: 4px solid var(--success-color);\n")
    cat("    margin-bottom: 20px;\n")
    cat("}\n\n")
    
    cat(".text-small { margin: 0; font-size: 0.9em; }\n")
    cat(".text-top { margin: 5px 0 0 0; }\n")
    cat(".text-top-large { margin: 10px 0 0 0; }\n")
    cat(".list-top { margin-top: 10px; }\n")
    cat(".list-small { margin: 5px 0; font-size: 0.9em; }\n\n")
    
    cat(".paper-link {\n")
    cat("    color: #74b9ff;\n")
    cat("    text-decoration: none;\n")
    cat("    font-weight: bold;\n")
    cat("    border: 1px solid #74b9ff;\n")
    cat("    padding: 5px 10px;\n")
    cat("    border-radius: 3px;\n")
    cat("    display: inline-block;\n")
    cat("}\n\n")
    
  } else {
    cat("\n✅ All files are already up to date!\n")
  }
}

# ============================================
# RUN THE SCRIPT
# ============================================

# Execute the main function
main()