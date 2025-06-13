# ============================================
# HTML FILES REFINEMENT SCRIPT
# Refines academic paper HTML files according to improved CSS
# ============================================

library(xml2)
library(stringr)
library(purrr)

# Function to improve formula boxes based on content length - ROBUST VERSION
improve_formula_boxes <- function(html_content) {
  # Find all formula divs using a more robust approach
  formula_matches <- str_locate_all(html_content, '<div class="formula">')[[1]]
  
  if (nrow(formula_matches) == 0) {
    return(html_content)
  }
  
  # Process from end to beginning to avoid position shifts
  for (i in nrow(formula_matches):1) {
    start_pos <- formula_matches[i, "start"]
    
    # Find the matching closing div
    temp_content <- substr(html_content, start_pos, nchar(html_content))
    
    # Simple approach: extract content between first > and </div>
    pattern <- '<div class="formula">(.*?)</div>'
    match <- str_match(temp_content, pattern)
    
    if (!is.na(match[1])) {
      formula_content <- match[2]
      content_length <- nchar(formula_content)
      
      # Determine appropriate class based on content length
      if (content_length < 50) {
        class_addition <- ' short'
      } else if (content_length < 120) {
        class_addition <- ' medium'
      } else {
        class_addition <- ' long'
      }
      
      # Check if formula contains line breaks or multiple equations
      if (str_detect(formula_content, "<br>|\\n|;|,.*=")) {
        class_addition <- paste0(class_addition, '" data-multiline="true')
      }
      
      # Replace the old formula div with improved version
      old_formula <- paste0('<div class="formula">', formula_content, '</div>')
      new_formula <- paste0('<div class="formula', class_addition, '">', formula_content, '</div>')
      
      html_content <- str_replace(html_content, fixed(old_formula), new_formula)
    }
  }
  
  return(html_content)
}

# Function to fix citation boxes
fix_citation_boxes <- function(html_content) {
  # Ensure citation boxes have proper structure
  html_content <- str_replace_all(html_content, 
                                  '<div class="citation">',
                                  '<div class="citation-box">')
  
  return(html_content)
}

# Function to improve step content boxes
improve_step_content <- function(html_content) {
  # Ensure step-details have proper spacing
  html_content <- str_replace_all(html_content, 
                                  '(<div class="step-details">\\s*)(.*?)(\\s*</div>)',
                                  function(match) {
                                    content <- str_match(match, '<div class="step-details">\\s*(.*?)\\s*</div>')[, 2]
                                    
                                    # Add proper paragraph structure if missing
                                    if (!str_detect(content, '<p>|<ul>|<li>|<strong>')) {
                                      content <- paste0('<p>', content, '</p>')
                                    }
                                    
                                    return(paste0('<div class="step-details">', content, '</div>'))
                                  })
  
  return(html_content)
}

# Function to enhance analysis cards
enhance_analysis_cards <- function(html_content) {
  # Improve analysis content structure
  html_content <- str_replace_all(html_content,
                                  '(<div class="analysis-content">\\s*)(.*?)(\\s*</div>)',
                                  function(match) {
                                    content <- str_match(match, '<div class="analysis-content">\\s*(.*?)\\s*</div>')[, 2]
                                    
                                    # Ensure proper paragraph structure
                                    if (!str_detect(content, '<p>')) {
                                      # Split by double line breaks and wrap in paragraphs
                                      paragraphs <- str_split(content, '\\n\\s*\\n')[[1]]
                                      content <- paste(map_chr(paragraphs, ~ paste0('<p>', .x, '</p>')), collapse = '\n')
                                    }
                                    
                                    return(paste0('<div class="analysis-content">', content, '</div>'))
                                  })
  
  return(html_content)
}

# Function to fix table containers - FIXED REGEX ISSUE
fix_table_containers <- function(html_content) {
  # Simple approach: wrap tables that aren't already in containers
  # First, find all existing table-container wrapped tables
  existing_wrapped <- str_extract_all(html_content, '<div class="table-container">.*?<table[^>]*>.*?</table>.*?</div>')[[1]]
  
  # Find all standalone tables
  all_tables <- str_extract_all(html_content, '<table[^>]*>.*?</table>')[[1]]
  
  # Process each table individually
  for (table in all_tables) {
    # Check if this table is already wrapped
    is_wrapped <- any(str_detect(existing_wrapped, fixed(table)))
    
    if (!is_wrapped) {
      # Wrap the unwrapped table
      wrapped_table <- paste0('<div class="table-container">', table, '</div>')
      html_content <- str_replace(html_content, fixed(table), wrapped_table)
    }
  }
  
  return(html_content)
}

# Function to improve content spacing - SIMPLIFIED
improve_content_spacing <- function(html_content) {
  # Add proper spacing between sections - simplified approach
  html_content <- str_replace_all(html_content,
                                  '</div>\\s*<div class="section">',
                                  '</div>\n        <div class="section">')
  
  # Ensure proper indentation for nested elements - simplified
  html_content <- str_replace_all(html_content,
                                  '(<div class="[^"]*">)\\s*(<div class="[^"]*">)',
                                  '\\1\n            \\2')
  
  return(html_content)
}

# Function to add missing IDs and improve navigation
add_navigation_improvements <- function(html_content) {
  # Ensure back-to-top functionality works
  html_content <- str_replace_all(html_content,
                                  '(<body[^>]*>)',
                                  '\\1')
  
  # Add section IDs for better navigation
  section_counter <- 0
  html_content <- str_replace_all(html_content,
                                  '<div class="section">',
                                  function(x) {
                                    section_counter <<- section_counter + 1
                                    paste0('<div class="section" id="section-', section_counter, '">')
                                  })
  
  return(html_content)
}

# Function to ensure proper meta tags
ensure_proper_meta <- function(html_content) {
  # Ensure charset and viewport are properly set
  if (!str_detect(html_content, 'charset="UTF-8"')) {
    html_content <- str_replace(html_content,
                                '<head>',
                                '<head>\n    <meta charset="UTF-8">')
  }
  
  if (!str_detect(html_content, 'name="viewport"')) {
    html_content <- str_replace(html_content,
                                '(<meta charset="UTF-8">)',
                                '\\1\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">')
  }
  
  return(html_content)
}

# Function to fix specific content issues - EXTENDED FOR ALL FILES
fix_content_specific_issues <- function(html_content, filename) {
  
  # Fix 2009_Garrick.html specific issues
  if (str_detect(filename, "2009_Garrick")) {
    # Improve deregression formulas
    html_content <- str_replace_all(html_content,
                                    'Deregressed EBV = EBV/r²',
                                    '<div class="formula short">Deregressed EBV = EBV/r²</div>')
    
    # Fix weighting formulas
    html_content <- str_replace_all(html_content,
                                    'w = \\(1-h²\\)/\\[c \\+ \\(1-r²\\)/r²\\]h²',
                                    '<div class="formula medium">w = (1-h²)/[c + (1-r²)/r²]h²</div>')
    
    # Fix basic model formulas
    html_content <- str_replace_all(html_content,
                                    'g = 1μ \\+ Ma \\+ ε',
                                    '<div class="formula short">g = 1μ + Ma + ε</div>')
    
    html_content <- str_replace_all(html_content,
                                    'y = Xb \\+ g \\+ e',
                                    '<div class="formula short">y = Xb + g + e</div>')
  }
  
  # Fix 1991_vanraden.html specific issues
  if (str_detect(filename, "1991_vanraden")) {
    # Improve animal model formula
    html_content <- str_replace_all(html_content,
                                    'y = Mm \\+ Za \\+ ZA₍ₘ₎g \\+ Pp \\+ Cc \\+ e',
                                    '<div class="formula long">y = Mm + Za + ZA<sub>m</sub>g + Pp + Cc + e</div>')
    
    # Fix YD formula
    html_content <- str_replace_all(html_content,
                                    'YD = yᵢ - \\(mᵢ \\+ pᵢ \\+ cᵢ\\)',
                                    '<div class="formula medium">YD = y<sub>i</sub> - (m<sub>i</sub> + p<sub>i</sub> + c<sub>i</sub>)</div>')
    
    # Fix PTA decomposition
    html_content <- str_replace_all(html_content,
                                    'PTA = w₁PA \\+ w₂\\(YD/2\\) \\+ w₃PC',
                                    '<div class="formula medium">PTA = w<sub>1</sub>PA + w<sub>2</sub>(YD/2) + w<sub>3</sub>PC</div>')
    
    # Fix reliability formulas
    html_content <- str_replace_all(html_content,
                                    'REL = r²\\(PTA, TA\\)',
                                    '<div class="formula short">REL = r²(PTA, TA)</div>')
    
    html_content <- str_replace_all(html_content,
                                    'REL = DE/\\(DE \\+ 14\\)',
                                    '<div class="formula short">REL = DE/(DE + 14)</div>')
  }
  
  # Fix 2006_bevova.html specific issues
  if (str_detect(filename, "2006_bevova")) {
    # This file appears to be mostly text-based with fewer formulas
    # Ensure proper citation box structure
    html_content <- str_replace_all(html_content,
                                    '<div class="ref-box">',
                                    '<div class="citation-box">')
  }
  
  # Fix 2008_vanraden.html specific issues
  if (str_detect(filename, "2008_vanraden")) {
    # Fix G-matrix formula
    html_content <- str_replace_all(html_content,
                                    'G = ZZ\' / \\[2∑pᵢ\\(1-pᵢ\\)\\]',
                                    '<div class="formula medium">G = ZZ\' / [2∑p<sub>i</sub>(1-p<sub>i</sub>)]</div>')
    
    # Fix method formulas
    html_content <- str_replace_all(html_content,
                                    'â = Z₂û where û solved from mixed model equations',
                                    '<div class="formula medium">â = Z<sub>2</sub>û where û solved from mixed model equations</div>')
    
    html_content <- str_replace_all(html_content,
                                    'â = G\\[G \\+ R\\(σₑ²/σₐ²\\)\\]⁻¹\\(y - Xb̂\\)',
                                    '<div class="formula long">â = G[G + R(σ<sub>e</sub>²/σ<sub>a</sub>²)]⁻¹(y - Xb̂)</div>')
    
    html_content <- str_replace_all(html_content,
                                    'â = \\[R⁻¹ \\+ G⁻¹\\(σₑ²/σₐ²\\)\\]⁻¹R⁻¹\\(y - Xb̂\\)',
                                    '<div class="formula long">â = [R⁻¹ + G⁻¹(σ<sub>e</sub>²/σ<sub>a</sub>²)]⁻¹R⁻¹(y - Xb̂)</div>')
  }
  
  # Fix 2009_chen.html specific issues
  if (str_detect(filename, "2009_chen")) {
    # Fix animal model formula
    html_content <- str_replace_all(html_content,
                                    'y = μ \\+ Za \\+ e',
                                    '<div class="formula short">y = μ + Za + e</div>')
    
    # Fix Mendelian sampling formulas
    html_content <- str_replace_all(html_content,
                                    'm̂ᵢ = âᵢ - ½\\(âₛ \\+ âᴅ\\)',
                                    '<div class="formula medium">m̂<sub>i</sub> = â<sub>i</sub> - ½(â<sub>s</sub> + â<sub>d</sub>)</div>')
    
    html_content <- str_replace_all(html_content,
                                    'm̂ = μ \\+ Wq \\+ ε',
                                    '<div class="formula short">m̂ = μ + Wq + ε</div>')
    
    html_content <- str_replace_all(html_content,
                                    'Wᵢ = gᵢ - ½\\(gₛ \\+ gᴅ\\)',
                                    '<div class="formula medium">W<sub>i</sub> = g<sub>i</sub> - ½(g<sub>s</sub> + g<sub>d</sub>)</div>')
    
    html_content <- str_replace_all(html_content,
                                    'âᵢ = ½\\(âₛ \\+ âᴅ\\) \\+ Ŵᵢq̂',
                                    '<div class="formula medium">â<sub>i</sub> = ½(â<sub>s</sub> + â<sub>d</sub>) + Ŵ<sub>i</sub>q̂</div>')
  }
  
  # Fix 2017_xue.html specific issues
  if (str_detect(filename, "2017_xue")) {
    # Improve model formulas
    html_content <- str_replace_all(html_content,
                                    'Y_ij = μ \\+ X_jk β_k \\+ E_i \\+ F_j \\+ ε_ijk',
                                    '<div class="formula medium">Y<sub>ij</sub> = μ + X<sub>jk</sub>β<sub>k</sub> + E<sub>i</sub> + F<sub>j</sub> + ε<sub>ijk</sub></div>')
    
    # Fix BLUE formulas
    html_content <- str_replace_all(html_content,
                                    'BLUE\\(Y_j\\) = μ \\+ F̂_j',
                                    '<div class="formula short">BLUE(Y<sub>j</sub>) = μ + F̂<sub>j</sub></div>')
    
    html_content <- str_replace_all(html_content,
                                    'BLUE\\(Y_j\\) = μ \\+ F_j \\+ X_jk β_k \\+ ε_jk',
                                    '<div class="formula medium">BLUE(Y<sub>j</sub>) = μ + F<sub>j</sub> + X<sub>jk</sub>β<sub>k</sub> + ε<sub>jk</sub></div>')
  }
  
  # Fix 2018_marees.html specific issues
  if (str_detect(filename, "2018_marees")) {
    # Improve PRS formula
    html_content <- str_replace_all(html_content,
                                    'PRS = Σ\\(Number of risk alleles × Weight\\)',
                                    '<div class="formula medium">PRS = Σ(Number of risk alleles × Weight)</div>')
  }
  
  # Fix 2018_oliveira.html specific issues
  if (str_detect(filename, "2018_oliveira")) {
    # Improve deregression formulas
    html_content <- str_replace_all(html_content,
                                    'dEBV_ij = PA_ij \\+ \\(EBV_ij - PA_ij\\) / R_i',
                                    '<div class="formula medium">dEBV<sub>ij</sub> = PA<sub>ij</sub> + (EBV<sub>ij</sub> - PA<sub>ij</sub>) / R<sub>i</sub></div>')
    
    # Fix adjustment formulas
    html_content <- str_replace_all(html_content,
                                    'PA_adj = PA_sire \\+ \\[VA × \\(PA_dam - MA\\)\\]',
                                    '<div class="formula medium">PA<sub>adj</sub> = PA<sub>sire</sub> + [VA × (PA<sub>dam</sub> - MA)]</div>')
  }
  
  # Fix 2019_stephan.html specific issues
  if (str_detect(filename, "2019_stephan")) {
    # Improve sweep formulas
    html_content <- str_replace_all(html_content,
                                    'π = π₀ × r/\\(r \\+ κan\\)',
                                    '<div class="formula medium">π = π<sub>0</sub> × r/(r + κan)</div>')
  }
  
  # Fix 2025_musa.html specific issues
  if (str_detect(filename, "2025_musa")) {
    # Improve similarity matrix formulas
    html_content <- str_replace_all(html_content,
                                    's<sub>i,j</sub> = \\|m<sub>i</sub>\'Rm<sub>j</sub>\\|',
                                    '<div class="formula medium">s<sub>i,j</sub> = |m<sub>i</sub>\'Rm<sub>j</sub>|</div>')
    
    html_content <- str_replace_all(html_content,
                                    'var\\(b<sub>i</sub>\\) = m<sub>i</sub>\'Rm<sub>i</sub>',
                                    '<div class="formula medium">var(b<sub>i</sub>) = m<sub>i</sub>\'Rm<sub>i</sub></div>')
    
    # Fix MOCS formulas
    html_content <- str_replace_all(html_content,
                                    'maximize: r<sub>t\\+1</sub> = n<sub>t</sub>\'r<sub>t</sub>',
                                    '<div class="formula medium" data-multiline="true">maximize: r<sub>t+1</sub> = n<sub>t</sub>\'r<sub>t</sub><br>subject to: Q<sub>t+1</sub> ≤ n<sub>t</sub>\'Q<sub>t</sub>n<sub>t</sub>/2</div>')
  }
  
  # Fix 2025_niehoff.html specific issues
  if (str_detect(filename, "2025_niehoff")) {
    # Fix MSV formulas
    html_content <- str_replace_all(html_content,
                                    'σ²ₘsᵥ = Σ pᵢ\\(1-pᵢ\\)α²ᵢ',
                                    '<div class="formula medium">σ²<sub>MSV</sub> = Σ p<sub>i</sub>(1-p<sub>i</sub>)α²<sub>i</sub></div>')
    
    html_content <- str_replace_all(html_content,
                                    'Index5 = BVₐₙᵢₘₐₗ \\+ xₘ × √\\(2×σ²ₘsᵥ\\)',
                                    '<div class="formula medium">Index5 = BV<sub>animal</sub> + x<sub>m</sub> × √(2×σ²<sub>MSV</sub>)</div>')
  }
  
  # Fix 2025_rochus.html specific issues
  if (str_detect(filename, "2025_rochus")) {
    # Fix Ne formulas
    html_content <- str_replace_all(html_content,
                                    'Ne = 4NmNf/\\(Nm \\+ Nf\\)',
                                    '<div class="formula medium">Ne = 4N<sub>m</sub>N<sub>f</sub>/(N<sub>m</sub> + N<sub>f</sub>)</div>')
  }
  
  return(html_content)
}

# Function to validate HTML structure
validate_html_structure <- function(html_content) {
  # Check for common issues
  issues <- list()
  
  # Check for unclosed tags
  open_divs <- str_count(html_content, '<div')
  close_divs <- str_count(html_content, '</div>')
  if (open_divs != close_divs) {
    issues <- append(issues, paste("Div mismatch: Open =", open_divs, "Close =", close_divs))
  }
  
  # Check for proper formula box structure
  formula_count <- str_count(html_content, '<div class="formula')
  if (formula_count > 0) {
    issues <- append(issues, paste("Found", formula_count, "formula boxes"))
  }
  
  return(issues)
}

# Main refinement function
refine_html_file <- function(input_file, output_file = NULL) {
  if (is.null(output_file)) {
    output_file <- input_file
  }
  
  cat("Processing:", input_file, "\n")
  
  # Read the file
  html_content <- readLines(input_file, warn = FALSE, encoding = "UTF-8")
  html_content <- paste(html_content, collapse = "\n")
  
  # Apply all improvements
  html_content <- ensure_proper_meta(html_content)
  html_content <- improve_formula_boxes(html_content)
  html_content <- fix_citation_boxes(html_content)
  html_content <- improve_step_content(html_content)
  html_content <- enhance_analysis_cards(html_content)
  html_content <- fix_table_containers(html_content)
  html_content <- improve_content_spacing(html_content)
  html_content <- add_navigation_improvements(html_content)
  html_content <- fix_content_specific_issues(html_content, input_file)
  
  # Validate structure
  issues <- validate_html_structure(html_content)
  if (length(issues) > 0) {
    cat("Issues found in", input_file, ":\n")
    for (issue in issues) {
      cat("  -", issue, "\n")
    }
  }
  
  # Write the refined file
  writeLines(html_content, output_file, useBytes = TRUE)
  cat("Completed:", output_file, "\n\n")
}

# Batch processing function
refine_all_html_files <- function(directory = ".", backup = TRUE) {
  # Find all HTML files
  html_files <- list.files(directory, pattern = "\\.html$", full.names = TRUE)
  
  cat("Found", length(html_files), "HTML files to process\n\n")
  
  for (file in html_files) {
    # Create backup if requested
    if (backup) {
      backup_file <- paste0(file, ".backup")
      file.copy(file, backup_file, overwrite = TRUE)
      cat("Backup created:", backup_file, "\n")
    }
    
    # Refine the file
    refine_html_file(file)
  }
  
  cat("All files processed successfully!\n")
}

# ============================================
# USAGE EXAMPLES
# ============================================

# Process a single file
# refine_html_file("2009_Garrick.html")

# Process all HTML files in current directory
# refine_all_html_files()

# Process specific files - UPDATED WITH ALL FILES
process_specific_files <- function() {
  files_to_process <- c(
    # Original 9 files
    "2009_Garrick.html",
    "2017_xue.html", 
    "2018_marees.html",
    "2018_oliveira.html",
    "2019_stephan.html",
    "2022_palma_vera.html",
    "2025_musa.html",
    "2025_niehoff.html",
    "2025_rochus.html",
    
    # Additional files
    "1991_vanraden.html",
    "2006_bevova.html",
    "2008_vanraden.html",
    "2009_chen.html"
  )
  
  cat("Processing", length(files_to_process), "HTML files:\n")
  
  for (file in files_to_process) {
    if (file.exists(file)) {
      refine_html_file(file)
    } else {
      cat("File not found:", file, "\n")
    }
  }
}

# ============================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================

# Function to preview changes before applying
preview_changes <- function(input_file) {
  html_content <- readLines(input_file, warn = FALSE, encoding = "UTF-8")
  html_content <- paste(html_content, collapse = "\n")
  
  original_formulas <- str_count(html_content, '<div class="formula">')
  
  # Apply improvements
  improved_content <- improve_formula_boxes(html_content)
  new_formulas <- str_count(improved_content, '<div class="formula')
  
  cat("Preview for", input_file, ":\n")
  cat("Original formula boxes:", original_formulas, "\n")
  cat("Improved formula boxes:", new_formulas, "\n")
  
  # Show some formula examples
  formulas <- str_extract_all(improved_content, '<div class="formula[^>]*">.*?</div>')[[1]]
  if (length(formulas) > 0) {
    cat("Example formulas:\n")
    for (i in 1:min(3, length(formulas))) {
      cat("  ", formulas[i], "\n")
    }
  }
}

# Function to check CSS compatibility
check_css_compatibility <- function(html_file, css_file = "style.css") {
  html_content <- readLines(html_file, warn = FALSE, encoding = "UTF-8")
  html_content <- paste(html_content, collapse = "\n")
  
  css_content <- readLines(css_file, warn = FALSE, encoding = "UTF-8")
  css_content <- paste(css_content, collapse = "\n")
  
  # Extract classes used in HTML
  html_classes <- str_extract_all(html_content, 'class="([^"]*)"')[[1]]
  html_classes <- str_replace_all(html_classes, 'class="', '')
  html_classes <- str_replace_all(html_classes, '"', '')
  html_classes <- unique(unlist(str_split(html_classes, "\\s+")))
  
  # Extract classes defined in CSS
  css_classes <- str_extract_all(css_content, '\\.([a-zA-Z0-9_-]+)')[[1]]
  css_classes <- str_replace_all(css_classes, '\\.', '')
  css_classes <- unique(css_classes)
  
  # Find missing classes
  missing_classes <- setdiff(html_classes, css_classes)
  
  cat("CSS Compatibility Check for", html_file, ":\n")
  cat("HTML classes:", length(html_classes), "\n")
  cat("CSS classes:", length(css_classes), "\n")
  cat("Missing in CSS:", length(missing_classes), "\n")
  
  if (length(missing_classes) > 0) {
    cat("Missing classes:\n")
    for (class in missing_classes) {
      cat("  -", class, "\n")
    }
  }
}

# ============================================
# EXECUTION
# ============================================

cat("HTML Refinement Script Loaded - COMPLETE VERSION\n")
cat("Available functions:\n")
cat("  - refine_html_file(file): Process single file\n")
cat("  - refine_all_html_files(): Process all HTML files\n")
cat("  - process_specific_files(): Process all 13 academic papers\n")
cat("  - preview_changes(file): Preview changes before applying\n")
cat("  - check_css_compatibility(file): Check CSS compatibility\n")
cat("\nFiles included:\n")
cat("  Original 9: 2009_Garrick, 2017_xue, 2018_marees, 2018_oliveira,\n")
cat("              2019_stephan, 2022_palma_vera, 2025_musa, 2025_niehoff, 2025_rochus\n")
cat("  Additional 4: 1991_vanraden, 2006_bevova, 2008_vanraden, 2009_chen\n")
cat("\nTo start processing all 13 files, run: process_specific_files()\n")

# Uncomment the next line to automatically process all files
# process_specific_files()