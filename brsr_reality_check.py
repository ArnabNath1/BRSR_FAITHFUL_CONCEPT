"""
BRSR Reality Check - Interactive Summary Page
Provides a comprehensive reality check of BRSR compliance and evidence quality
"""

import json
import numpy as np
from collections import defaultdict
from datetime import datetime


class BRSRRealityCheck:
    """
    Creates a comprehensive BRSR reality check summary
    """
    
    def __init__(self, concept_map_json: str):
        """
        Initialize with the concept map JSON file
        
        Args:
            concept_map_json: Path to the concept_map.json file
        """
        with open(concept_map_json, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.concepts = self.data['concepts']
        self.relations = self.data['relations']
        
        # BRSR Principles
        self.brsr_principles = {
            'P1': 'Ethics, Transparency and Accountability',
            'P2': 'Product Lifecycle Sustainability',
            'P3': 'Employee Well-being',
            'P4': 'Stakeholder Engagement',
            'P5': 'Human Rights',
            'P6': 'Environment',
            'P7': 'Policy Advocacy',
            'P8': 'Inclusive Growth',
            'P9': 'Customer Value'
        }
        
        # ESG Categories
        self.esg_categories = ['Environmental', 'Social', 'Governance']
    
    def map_concept_to_principle(self, concept_text: str) -> str:
        """Map a concept to a BRSR principle"""
        concept_lower = concept_text.lower()
        
        if any(kw in concept_lower for kw in ['environment', 'climate', 'carbon', 'emission', 'energy', 'renewable', 'waste', 'water', 'ghg']):
            return 'P6'
        elif any(kw in concept_lower for kw in ['employee', 'workforce', 'talent', 'retention', 'diversity', 'inclusion', 'health', 'safety', 'welfare']):
            return 'P3'
        elif any(kw in concept_lower for kw in ['governance', 'ethics', 'compliance', 'transparency', 'accountability', 'disclosure', 'board']):
            return 'P1'
        elif any(kw in concept_lower for kw in ['stakeholder', 'community', 'engagement', 'consultation', 'feedback']):
            return 'P4'
        elif any(kw in concept_lower for kw in ['social impact', 'community development', 'inclusive', 'growth', 'csr']):
            return 'P8'
        elif any(kw in concept_lower for kw in ['product', 'service', 'lifecycle', 'circular', 'epr']):
            return 'P2'
        elif any(kw in concept_lower for kw in ['human rights', 'labor', 'equality', 'discrimination']):
            return 'P5'
        elif any(kw in concept_lower for kw in ['policy', 'advocacy', 'regulation', 'standard', 'framework']):
            return 'P7'
        elif any(kw in concept_lower for kw in ['customer', 'client', 'value', 'satisfaction', 'quality']):
            return 'P9'
        return 'P1'
    
    def categorize_concept(self, concept_text: str) -> str:
        """Categorize a concept into ESG categories"""
        concept_lower = concept_text.lower()
        
        if any(kw in concept_lower for kw in ['environment', 'climate', 'carbon', 'emission', 'energy', 'renewable', 'waste', 'water', 'sustainability']):
            return 'Environmental'
        elif any(kw in concept_lower for kw in ['employee', 'social', 'community', 'diversity', 'inclusion', 'health', 'safety', 'human rights', 'stakeholder', 'welfare']):
            return 'Social'
        elif any(kw in concept_lower for kw in ['governance', 'ethics', 'compliance', 'transparency', 'accountability', 'board', 'management', 'policy', 'disclosure']):
            return 'Governance'
        return 'Other'
    
    def analyze_brsr_coverage(self):
        """Analyze BRSR principle coverage"""
        principle_coverage = defaultdict(lambda: {'count': 0, 'confidence': [], 'concepts': []})
        
        for concept in self.concepts:
            if concept['confidence'] >= 0.6:
                principle = self.map_concept_to_principle(concept['text'])
                principle_coverage[principle]['count'] += 1
                principle_coverage[principle]['confidence'].append(concept['confidence'])
                principle_coverage[principle]['concepts'].append(concept['text'])
        
        return principle_coverage
    
    def analyze_esg_coverage(self):
        """Analyze ESG category coverage"""
        esg_coverage = defaultdict(lambda: {'count': 0, 'confidence': [], 'concepts': []})
        
        for concept in self.concepts:
            if concept['confidence'] >= 0.6:
                category = self.categorize_concept(concept['text'])
                esg_coverage[category]['count'] += 1
                esg_coverage[category]['confidence'].append(concept['confidence'])
                esg_coverage[category]['concepts'].append(concept['text'])
        
        return esg_coverage
    
    def calculate_reality_score(self):
        """Calculate overall BRSR reality score"""
        scores = {
            'coverage': 0,
            'confidence': 0,
            'evidence': 0,
            'balance': 0
        }
        
        # Coverage score (how many principles covered)
        principle_coverage = self.analyze_brsr_coverage()
        covered_principles = len([p for p in principle_coverage if principle_coverage[p]['count'] > 0])
        scores['coverage'] = (covered_principles / 9) * 100
        
        # Confidence score (average confidence)
        all_confidences = [c['confidence'] for c in self.concepts if c['confidence'] >= 0.6]
        scores['confidence'] = (np.mean(all_confidences) if all_confidences else 0) * 100
        
        # Evidence score (concepts with page references)
        concepts_with_evidence = len([c for c in self.concepts if c['page_number'] > 0])
        scores['evidence'] = (concepts_with_evidence / len(self.concepts)) * 100
        
        # Balance score (distribution across ESG)
        esg_coverage = self.analyze_esg_coverage()
        esg_counts = [esg_coverage[cat]['count'] for cat in self.esg_categories]
        if max(esg_counts) > 0:
            balance = 1 - (np.std(esg_counts) / np.mean(esg_counts))
            scores['balance'] = max(0, balance * 100)
        
        # Overall score
        overall = np.mean(list(scores.values()))
        
        return scores, overall
    
    def get_grade(self, score):
        """Get letter grade based on score"""
        if score >= 90:
            return 'A+', '🟢', 'Excellent'
        elif score >= 80:
            return 'A', '🟢', 'Very Good'
        elif score >= 70:
            return 'B+', '🟡', 'Good'
        elif score >= 60:
            return 'B', '🟡', 'Satisfactory'
        elif score >= 50:
            return 'C', '🟠', 'Needs Improvement'
        else:
            return 'D', '🔴', 'Poor'
    
    def generate_html_summary(self, output_path: str = "brsr_reality_check.html"):
        """Generate interactive HTML summary page"""
        
        principle_coverage = self.analyze_brsr_coverage()
        esg_coverage = self.analyze_esg_coverage()
        scores, overall_score = self.calculate_reality_score()
        grade, indicator, assessment = self.get_grade(overall_score)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BRSR Reality Check - Sustainability Report Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .timestamp {{
            background: rgba(255,255,255,0.1);
            padding: 10px 20px;
            border-radius: 20px;
            display: inline-block;
            margin-top: 15px;
            font-size: 0.9em;
        }}
        
        .overall-score {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
        }}
        
        .score-circle {{
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: white;
            margin: 20px auto;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .score-value {{
            font-size: 4em;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .score-grade {{
            font-size: 1.5em;
            color: #7f8c8d;
            margin-top: -10px;
        }}
        
        .assessment {{
            font-size: 2em;
            margin-top: 20px;
            font-weight: bold;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .metric-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .metric-bar {{
            height: 10px;
            background: #ecf0f1;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 15px;
        }}
        
        .metric-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 5px;
            transition: width 1s ease-out;
        }}
        
        .section {{
            padding: 40px;
        }}
        
        .section-title {{
            font-size: 2em;
            color: #2c3e50;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }}
        
        .principle-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .principle-card {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 12px;
            border-left: 5px solid #667eea;
        }}
        
        .principle-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .principle-code {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .principle-count {{
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        
        .principle-name {{
            font-size: 1.1em;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .confidence-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        
        .confidence-high {{
            background: #2ecc71;
            color: white;
        }}
        
        .confidence-medium {{
            background: #f39c12;
            color: white;
        }}
        
        .confidence-low {{
            background: #e74c3c;
            color: white;
        }}
        
        .esg-section {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        
        .esg-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 30px;
            margin-top: 20px;
        }}
        
        .esg-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .esg-icon {{
            font-size: 4em;
            margin-bottom: 15px;
        }}
        
        .esg-title {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        
        .esg-count {{
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .key-findings {{
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .key-findings h3 {{
            color: #856404;
            margin-bottom: 15px;
        }}
        
        .key-findings ul {{
            list-style-position: inside;
            color: #856404;
        }}
        
        .key-findings li {{
            margin: 10px 0;
            line-height: 1.6;
        }}
        
        .recommendations {{
            background: #d1ecf1;
            border-left: 5px solid #17a2b8;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .recommendations h3 {{
            color: #0c5460;
            margin-bottom: 15px;
        }}
        
        .recommendations ul {{
            list-style-position: inside;
            color: #0c5460;
        }}
        
        .recommendations li {{
            margin: 10px 0;
            line-height: 1.6;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .stats-row {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin: 30px 0;
        }}
        
        .stat-item {{
            text-align: center;
            padding: 20px;
        }}
        
        .stat-number {{
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            font-size: 1em;
            color: #7f8c8d;
            margin-top: 10px;
        }}
        
        @media (max-width: 768px) {{
            .esg-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .score-circle {{
                width: 150px;
                height: 150px;
            }}
            
            .score-value {{
                font-size: 3em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🎯 BRSR Reality Check</h1>
            <p>Business Responsibility & Sustainability Reporting Analysis</p>
            <div class="timestamp">
                📅 Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </div>
        </div>
        
        <!-- Overall Score -->
        <div class="overall-score">
            <h2>Overall BRSR Compliance Score</h2>
            <div class="score-circle">
                <div class="score-value">{overall_score:.0f}</div>
                <div class="score-grade">Grade: {grade}</div>
            </div>
            <div class="assessment">{indicator} {assessment}</div>
        </div>
        
        <!-- Metrics Grid -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Coverage Score</div>
                <div class="metric-value">{scores['coverage']:.0f}%</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {scores['coverage']}%"></div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Confidence Score</div>
                <div class="metric-value">{scores['confidence']:.0f}%</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {scores['confidence']}%"></div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Evidence Score</div>
                <div class="metric-value">{scores['evidence']:.0f}%</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {scores['evidence']}%"></div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Balance Score</div>
                <div class="metric-value">{scores['balance']:.0f}%</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: {scores['balance']}%"></div>
                </div>
            </div>
        </div>
        
        <!-- Statistics Row -->
        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-number">{len(self.concepts)}</div>
                <div class="stat-label">Total Concepts</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(self.relations)}</div>
                <div class="stat-label">Relationships</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len([c for c in self.concepts if c['confidence'] >= 0.7])}</div>
                <div class="stat-label">High Confidence</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{max([c['page_number'] for c in self.concepts])}</div>
                <div class="stat-label">Pages Analyzed</div>
            </div>
        </div>
        
        <!-- ESG Coverage -->
        <div class="section esg-section">
            <h2 class="section-title">ESG Category Coverage</h2>
            <div class="esg-grid">
                <div class="esg-card">
                    <div class="esg-icon">🌍</div>
                    <div class="esg-title">Environmental</div>
                    <div class="esg-count">{esg_coverage['Environmental']['count']}</div>
                    <div class="confidence-badge confidence-{('high' if np.mean(esg_coverage['Environmental']['confidence']) >= 0.7 else 'medium') if esg_coverage['Environmental']['confidence'] else 'low'}">
                        Avg: {np.mean(esg_coverage['Environmental']['confidence']) * 100 if esg_coverage['Environmental']['confidence'] else 0:.0f}%
                    </div>
                </div>
                
                <div class="esg-card">
                    <div class="esg-icon">👥</div>
                    <div class="esg-title">Social</div>
                    <div class="esg-count">{esg_coverage['Social']['count']}</div>
                    <div class="confidence-badge confidence-{('high' if np.mean(esg_coverage['Social']['confidence']) >= 0.7 else 'medium') if esg_coverage['Social']['confidence'] else 'low'}">
                        Avg: {np.mean(esg_coverage['Social']['confidence']) * 100 if esg_coverage['Social']['confidence'] else 0:.0f}%
                    </div>
                </div>
                
                <div class="esg-card">
                    <div class="esg-icon">⚖️</div>
                    <div class="esg-title">Governance</div>
                    <div class="esg-count">{esg_coverage['Governance']['count']}</div>
                    <div class="confidence-badge confidence-{('high' if np.mean(esg_coverage['Governance']['confidence']) >= 0.7 else 'medium') if esg_coverage['Governance']['confidence'] else 'low'}">
                        Avg: {np.mean(esg_coverage['Governance']['confidence']) * 100 if esg_coverage['Governance']['confidence'] else 0:.0f}%
                    </div>
                </div>
            </div>
        </div>
        
        <!-- BRSR Principles Coverage -->
        <div class="section">
            <h2 class="section-title">BRSR Principles Coverage (9 Principles)</h2>
            <div class="principle-grid">
"""
        
        # Add principle cards
        for code in sorted(self.brsr_principles.keys()):
            name = self.brsr_principles[code]
            coverage = principle_coverage[code]
            count = coverage['count']
            avg_conf = np.mean(coverage['confidence']) if coverage['confidence'] else 0
            conf_class = 'high' if avg_conf >= 0.7 else ('medium' if avg_conf >= 0.5 else 'low')
            
            html += f"""
                <div class="principle-card">
                    <div class="principle-header">
                        <div class="principle-code">{code}</div>
                        <div class="principle-count">{count}</div>
                    </div>
                    <div class="principle-name">{name}</div>
                    <div class="confidence-badge confidence-{conf_class}">
                        Confidence: {avg_conf * 100:.0f}%
                    </div>
                </div>
"""
        
        # Key findings
        top_principle = max(principle_coverage.items(), key=lambda x: x[1]['count'])
        weakest_principle = min(principle_coverage.items(), key=lambda x: x[1]['count'])
        
        html += f"""
            </div>
        </div>
        
        <!-- Key Findings -->
        <div class="section">
            <div class="key-findings">
                <h3>🔍 Key Findings</h3>
                <ul>
                    <li><strong>Strongest Coverage:</strong> {top_principle[0]} - {self.brsr_principles[top_principle[0]]} ({top_principle[1]['count']} concepts)</li>
                    <li><strong>Weakest Coverage:</strong> {weakest_principle[0]} - {self.brsr_principles[weakest_principle[0]]} ({weakest_principle[1]['count']} concepts)</li>
                    <li><strong>Overall Confidence:</strong> {np.mean([c['confidence'] for c in self.concepts]) * 100:.1f}% average across all concepts</li>
                    <li><strong>Evidence Quality:</strong> {len([c for c in self.concepts if c['page_number'] > 0])} out of {len(self.concepts)} concepts have page references</li>
                    <li><strong>ESG Balance:</strong> {'Well-balanced' if scores['balance'] >= 70 else 'Needs better balance'} distribution across Environmental, Social, and Governance</li>
                </ul>
            </div>
            
            <div class="recommendations">
                <h3>💡 Recommendations</h3>
                <ul>
"""
        
        # Generate recommendations
        if scores['coverage'] < 80:
            html += "                    <li>Expand coverage of underrepresented BRSR principles, particularly " + weakest_principle[0] + "</li>\n"
        
        if scores['confidence'] < 70:
            html += "                    <li>Improve confidence scores by providing more specific and detailed disclosures</li>\n"
        
        if scores['balance'] < 70:
            html += "                    <li>Balance ESG reporting to ensure comprehensive coverage across all three pillars</li>\n"
        
        if esg_coverage['Environmental']['count'] > esg_coverage['Social']['count'] * 2:
            html += "                    <li>Strengthen social responsibility disclosures to match environmental reporting depth</li>\n"
        
        html += """
                    <li>Maintain strong evidence-based reporting with clear page references and context</li>
                    <li>Continue tracking concept drift to monitor reporting evolution over time</li>
                </ul>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p><strong>Faithful Concept Mapper</strong> - GenAI Intern Evaluation</p>
            <p style="margin-top: 10px; opacity: 0.8;">
                This analysis is based on automated concept extraction and mapping.<br>
                All concepts are faithfully extracted from the source document with full traceability.
            </p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"BRSR Reality Check saved to {output_path}")
        
        return overall_score, scores


def main():
    """Main function to generate BRSR Reality Check"""
    
    print("="*80)
    print("GENERATING BRSR REALITY CHECK")
    print("="*80)
    
    # Initialize reality check
    reality_check = BRSRRealityCheck("d:/ArcTechnologies/concept_map.json")
    
    # Generate HTML summary
    print("\nCreating interactive HTML summary...")
    overall_score, scores = reality_check.generate_html_summary("d:/ArcTechnologies/brsr_reality_check.html")
    
    # Print summary
    print("\n" + "="*80)
    print("BRSR REALITY CHECK SUMMARY")
    print("="*80)
    print(f"\nOverall Score: {overall_score:.1f}/100")
    print(f"\nDetailed Scores:")
    print(f"  Coverage Score:    {scores['coverage']:.1f}%")
    print(f"  Confidence Score:  {scores['confidence']:.1f}%")
    print(f"  Evidence Score:    {scores['evidence']:.1f}%")
    print(f"  Balance Score:     {scores['balance']:.1f}%")
    
    grade, indicator, assessment = reality_check.get_grade(overall_score)
    print(f"\nGrade: {grade} - {assessment} {indicator}")
    
    print("\n" + "="*80)
    print("REALITY CHECK COMPLETE!")
    print("="*80)
    print("\nGenerated file:")
    print("  - brsr_reality_check.html: Interactive on-page summary")
    print("\nOpen the HTML file in a web browser to view the complete reality check.")


if __name__ == "__main__":
    main()
