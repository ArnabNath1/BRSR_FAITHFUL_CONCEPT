# 📊 BRSR Analysis - Complete Visualization Suite

## 🎯 Overview

This project provides a comprehensive analysis of the Business Responsibility and Sustainability Report (BRSR) with multiple interactive visualizations and dashboards.

---

## 📁 Interactive Dashboards (Open in Browser)

### 1. **BRSR Reality Check** ⭐ NEW
**File:** `brsr_reality_check.html`

**What it shows:**
- 🎯 Overall BRSR Compliance Score (0-100)
- 📊 Coverage, Confidence, Evidence, and Balance scores
- 🌍 ESG Category breakdown (Environmental, Social, Governance)
- 📋 All 9 BRSR Principles coverage analysis
- 🔍 Key findings and recommendations
- 📈 Visual metrics with color-coded indicators

**Best for:** Quick assessment of BRSR compliance quality

---

### 2. **Sankey Diagrams**
**Files:** 
- `brsr_sankey.html` - Main flow diagram
- `brsr_detailed_sankey.html` - Detailed with top concepts

**What it shows:**
- Flow from BRSR Principles → ESG Categories → Evidence
- Color-coded flows by principle
- Interactive hover to see exact numbers
- Visual representation of evidence distribution

**Best for:** Understanding how principles map to evidence

---

### 3. **Drift Dashboard**
**Files:**
- `drift_dashboard.html` - 6-panel comprehensive dashboard
- `category_drift_dashboard.html` - ESG category evolution

**What it shows:**
- 📈 Concept count evolution by page
- 📊 Confidence trends across document
- 🎨 Color-coded confidence drift (🟢 stable, 🔵 positive, 🔴 negative)
- 🔍 New concept discovery patterns
- 🔗 Relationship density analysis
- 🌡️ Drift heatmap

**Best for:** Tracking how concepts evolve through the document

---

## 📊 Static Visualizations

### 4. **Network Graph**
**File:** `concept_map.png`

**What it shows:**
- 584 concepts as nodes
- 590 relationships as edges
- Node size = confidence level
- Edge width = relationship strength

**Best for:** Understanding concept interconnections

---

### 5. **Sankey Diagram (Static)**
**File:** `brsr_sankey.png`

**What it shows:**
- Static version of the Sankey flow
- BRSR Principles to Evidence mapping

**Best for:** Reports and presentations

---

## 📄 Data Files

### 6. **JSON Export**
**File:** `concept_map.json` (588 KB)

**Contains:**
- All 584 concepts with metadata
- All 590 relationships with evidence
- Page references and context
- Confidence scores

**Best for:** Programmatic analysis and custom visualizations

---

### 7. **Text Reports**
**Files:**
- `concept_map_report.txt` - Main analysis report
- `drift_report.txt` - Drift analysis report

**Contains:**
- Summary statistics
- Top concepts by confidence
- Relationship type distribution
- Sample relationships with evidence
- Drift analysis findings

**Best for:** Quick text-based review

---

## 🚀 Quick Start Guide

### View the Reality Check (Recommended First Step)
```bash
# Open in your default browser
start brsr_reality_check.html
```

### Explore Interactive Dashboards
```bash
# Sankey diagram
start brsr_sankey.html

# Drift dashboard
start drift_dashboard.html

# Category drift
start category_drift_dashboard.html
```

### Regenerate Visualizations
```bash
# Install dependencies (if needed)
pip install -r requirements.txt

# Run individual generators
python brsr_reality_check.py
python sankey_visualizer.py
python drift_dashboard.py

# Or run the main concept mapper
python concept_mapper.py
```

---

## 📊 Scoring Breakdown

### BRSR Reality Check Scores

**Coverage Score** (0-100%)
- Measures how many of the 9 BRSR principles are covered
- Higher = more comprehensive coverage

**Confidence Score** (0-100%)
- Average confidence of extracted concepts
- Higher = more reliable evidence

**Evidence Score** (0-100%)
- Percentage of concepts with page references
- Higher = better traceability

**Balance Score** (0-100%)
- Distribution across ESG categories
- Higher = more balanced reporting

**Overall Score**
- Average of all four scores
- Graded: A+ (90+), A (80+), B+ (70+), B (60+), C (50+), D (<50)

---

## 🎨 Color Coding Guide

### Confidence Levels
- 🟢 **Green (High)**: ≥70% confidence
- 🟡 **Yellow (Medium)**: 50-69% confidence
- 🔴 **Red (Low)**: <50% confidence

### Drift Indicators
- 🟢 **Green**: Stable (drift < 0.05)
- 🔵 **Blue**: Positive drift (increasing)
- 🔴 **Red**: Negative drift (decreasing)

### ESG Categories
- 🌍 **Green**: Environmental
- 👥 **Blue**: Social
- ⚖️ **Red**: Governance

---

## 📈 Key Metrics Summary

| Metric | Value |
|--------|-------|
| **Total Concepts** | 584 |
| **Total Relationships** | 590 |
| **Average Confidence** | 73.1% |
| **Pages Analyzed** | 40 |
| **High Confidence Concepts** | 420+ |
| **BRSR Principles Covered** | 9/9 |
| **ESG Categories** | 3 (E, S, G) |

---

## 🔍 What Each Visualization Tells You

### BRSR Reality Check
**Question:** "How well does this report comply with BRSR standards?"
**Answer:** Overall score, principle coverage, and specific gaps

### Sankey Diagram
**Question:** "How do BRSR principles flow to actual evidence?"
**Answer:** Visual flow showing principle → category → evidence mapping

### Drift Dashboard
**Question:** "How do concepts evolve through the document?"
**Answer:** Trends in concept introduction, confidence, and relationships

### Network Graph
**Question:** "How are concepts interconnected?"
**Answer:** Visual network showing all relationships

---

## 💡 Use Cases

### For Auditors
1. Start with **BRSR Reality Check** for overall assessment
2. Review **Sankey Diagram** for evidence mapping
3. Check **Drift Dashboard** for consistency

### For Report Writers
1. Use **BRSR Reality Check** to identify gaps
2. Review **Network Graph** for concept relationships
3. Check **Drift Dashboard** for balanced coverage

### For Analysts
1. Export **JSON data** for custom analysis
2. Review **Text Reports** for detailed findings
3. Use **Drift Dashboard** for trend analysis

---

## 🎯 Next Steps

### To Improve BRSR Compliance
1. Review recommendations in **BRSR Reality Check**
2. Focus on underrepresented principles
3. Balance ESG category coverage
4. Increase evidence quality and traceability

### To Explore Further
1. Open **Interactive Dashboards** in browser
2. Hover over elements for detailed information
3. Review **JSON data** for raw concept information
4. Read **Text Reports** for comprehensive analysis

---

## 📞 Support

### Files Overview
- **Python Scripts**: `*.py` - Generators for visualizations
- **HTML Files**: `*.html` - Interactive dashboards
- **Data Files**: `*.json`, `*.txt` - Raw data and reports
- **Images**: `*.png` - Static visualizations
- **Documentation**: `*.md` - Guides and summaries

### Regeneration
All visualizations can be regenerated by running the respective Python scripts. The source data (`concept_map.json`) is the foundation for all visualizations.

---

## ✅ Checklist

- [ ] Viewed BRSR Reality Check
- [ ] Explored Sankey Diagram
- [ ] Reviewed Drift Dashboard
- [ ] Examined Network Graph
- [ ] Read Text Reports
- [ ] Reviewed JSON Data
- [ ] Understood Scoring System
- [ ] Identified Key Findings
- [ ] Noted Recommendations

---

**Generated:** December 5, 2025  
**Project:** Faithful Concept Mapper - GenAI Intern Evaluation  
**Status:** ✅ Complete

---

*All visualizations are interactive and can be opened directly in a web browser. No server required.*
