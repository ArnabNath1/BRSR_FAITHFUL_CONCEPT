"""
Sankey Diagram Visualizer for BRSR Principles to Evidence Flows
Creates a Sankey diagram showing how BRSR principles map to evidence in the document
"""

import json
import plotly.graph_objects as go
from collections import defaultdict
import re


class BRSRSankeyVisualizer:
    """
    Creates Sankey diagrams showing flows from BRSR principles to evidence
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
        
        # BRSR Principles (9 principles from NGRBC)
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
        self.esg_categories = {
            'Environmental': ['environment', 'climate', 'carbon', 'emission', 'energy', 
                            'renewable', 'waste', 'water', 'pollution', 'sustainability'],
            'Social': ['employee', 'social', 'community', 'diversity', 'inclusion', 
                      'health', 'safety', 'human rights', 'stakeholder', 'welfare'],
            'Governance': ['governance', 'ethics', 'compliance', 'transparency', 
                          'accountability', 'board', 'management', 'policy', 'disclosure']
        }
    
    def categorize_concept(self, concept_text: str) -> str:
        """
        Categorize a concept into ESG categories
        
        Args:
            concept_text: The concept text
            
        Returns:
            ESG category name
        """
        concept_lower = concept_text.lower()
        
        for category, keywords in self.esg_categories.items():
            if any(keyword in concept_lower for keyword in keywords):
                return category
        
        return 'Other'
    
    def map_concept_to_principle(self, concept_text: str) -> str:
        """
        Map a concept to a BRSR principle
        
        Args:
            concept_text: The concept text
            
        Returns:
            BRSR principle code (P1-P9)
        """
        concept_lower = concept_text.lower()
        
        # P6: Environment
        if any(kw in concept_lower for kw in ['environment', 'climate', 'carbon', 
                                                'emission', 'energy', 'renewable', 
                                                'waste', 'water', 'ghg']):
            return 'P6'
        
        # P3: Employee Well-being
        if any(kw in concept_lower for kw in ['employee', 'workforce', 'talent', 
                                                'retention', 'diversity', 'inclusion',
                                                'health', 'safety', 'welfare']):
            return 'P3'
        
        # P1: Ethics, Transparency and Accountability
        if any(kw in concept_lower for kw in ['governance', 'ethics', 'compliance', 
                                                'transparency', 'accountability', 
                                                'disclosure', 'board']):
            return 'P1'
        
        # P4: Stakeholder Engagement
        if any(kw in concept_lower for kw in ['stakeholder', 'community', 'engagement',
                                                'consultation', 'feedback']):
            return 'P4'
        
        # P8: Inclusive Growth
        if any(kw in concept_lower for kw in ['social impact', 'community development',
                                                'inclusive', 'growth', 'csr']):
            return 'P8'
        
        # P2: Product Lifecycle Sustainability
        if any(kw in concept_lower for kw in ['product', 'service', 'lifecycle', 
                                                'circular', 'epr']):
            return 'P2'
        
        # P5: Human Rights
        if any(kw in concept_lower for kw in ['human rights', 'labor', 'equality',
                                                'discrimination']):
            return 'P5'
        
        # P7: Policy Advocacy
        if any(kw in concept_lower for kw in ['policy', 'advocacy', 'regulation',
                                                'standard', 'framework']):
            return 'P7'
        
        # P9: Customer Value
        if any(kw in concept_lower for kw in ['customer', 'client', 'value', 
                                                'satisfaction', 'quality']):
            return 'P9'
        
        return 'P1'  # Default to P1
    
    def create_principle_to_evidence_sankey(self, output_path: str = "brsr_sankey.html"):
        """
        Create a Sankey diagram showing BRSR Principles → ESG Categories → Evidence
        
        Args:
            output_path: Path to save the HTML file
        """
        # Count flows
        principle_to_esg = defaultdict(lambda: defaultdict(int))
        esg_to_evidence = defaultdict(lambda: defaultdict(int))
        
        # Process concepts
        for concept in self.concepts:
            if concept['confidence'] >= 0.7:  # Only high-confidence concepts
                principle = self.map_concept_to_principle(concept['text'])
                esg_category = self.categorize_concept(concept['text'])
                
                # Flow: Principle → ESG Category
                principle_to_esg[principle][esg_category] += 1
                
                # Flow: ESG Category → Evidence Type
                if concept['page_number'] > 0:
                    evidence_type = f"Page {concept['page_number']}"
                    esg_to_evidence[esg_category]['Document Evidence'] += 1
        
        # Build Sankey data
        labels = []
        sources = []
        targets = []
        values = []
        colors = []
        
        # Color schemes
        principle_colors = {
            'P1': 'rgba(31, 119, 180, 0.8)',
            'P2': 'rgba(255, 127, 14, 0.8)',
            'P3': 'rgba(44, 160, 44, 0.8)',
            'P4': 'rgba(214, 39, 40, 0.8)',
            'P5': 'rgba(148, 103, 189, 0.8)',
            'P6': 'rgba(140, 86, 75, 0.8)',
            'P7': 'rgba(227, 119, 194, 0.8)',
            'P8': 'rgba(127, 127, 127, 0.8)',
            'P9': 'rgba(188, 189, 34, 0.8)'
        }
        
        esg_colors = {
            'Environmental': 'rgba(34, 139, 34, 0.8)',
            'Social': 'rgba(65, 105, 225, 0.8)',
            'Governance': 'rgba(220, 20, 60, 0.8)',
            'Other': 'rgba(128, 128, 128, 0.8)'
        }
        
        # Add BRSR Principles as nodes
        principle_indices = {}
        for code, name in self.brsr_principles.items():
            principle_indices[code] = len(labels)
            labels.append(f"{code}: {name}")
        
        # Add ESG Categories as nodes
        esg_indices = {}
        for category in ['Environmental', 'Social', 'Governance', 'Other']:
            esg_indices[category] = len(labels)
            labels.append(f"{category}")
        
        # Add Evidence node
        evidence_index = len(labels)
        labels.append("Document Evidence")
        
        # Create flows: Principle → ESG Category
        for principle, esg_flows in principle_to_esg.items():
            for esg_category, count in esg_flows.items():
                if count > 0:
                    sources.append(principle_indices[principle])
                    targets.append(esg_indices[esg_category])
                    values.append(count)
                    colors.append(principle_colors.get(principle, 'rgba(128, 128, 128, 0.4)'))
        
        # Create flows: ESG Category → Evidence
        for esg_category, evidence_flows in esg_to_evidence.items():
            for evidence_type, count in evidence_flows.items():
                if count > 0:
                    sources.append(esg_indices[esg_category])
                    targets.append(evidence_index)
                    values.append(count)
                    colors.append(esg_colors.get(esg_category, 'rgba(128, 128, 128, 0.4)'))
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=[
                    # BRSR Principles
                    principle_colors['P1'], principle_colors['P2'], principle_colors['P3'],
                    principle_colors['P4'], principle_colors['P5'], principle_colors['P6'],
                    principle_colors['P7'], principle_colors['P8'], principle_colors['P9'],
                    # ESG Categories
                    esg_colors['Environmental'], esg_colors['Social'], 
                    esg_colors['Governance'], esg_colors['Other'],
                    # Evidence
                    'rgba(255, 215, 0, 0.8)'
                ]
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors
            )
        )])
        
        fig.update_layout(
            title={
                'text': "BRSR Principles to Evidence Flow<br><sub>Business Responsibility and Sustainability Reporting</sub>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#2c3e50'}
            },
            font=dict(size=12, family="Arial, sans-serif"),
            height=800,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            paper_bgcolor='white'
        )
        
        # Save to HTML
        fig.write_html(output_path)
        print(f"Sankey diagram saved to {output_path}")
        
        # Also save as static image
        try:
            fig.write_image(output_path.replace('.html', '.png'), width=1400, height=800)
            print(f"Static image saved to {output_path.replace('.html', '.png')}")
        except Exception as e:
            print(f"Note: Could not save static image (install kaleido for PNG export): {e}")
        
        return fig
    
    def create_detailed_sankey(self, output_path: str = "brsr_detailed_sankey.html"):
        """
        Create a detailed Sankey diagram with more granular flows
        
        Args:
            output_path: Path to save the HTML file
        """
        # This creates: Principles → ESG → Top Concepts → Evidence Pages
        
        labels = []
        sources = []
        targets = []
        values = []
        
        # Add BRSR Principles
        principle_indices = {}
        for code, name in self.brsr_principles.items():
            principle_indices[code] = len(labels)
            labels.append(f"{code}")
        
        # Add ESG Categories
        esg_indices = {}
        for category in ['Environmental', 'Social', 'Governance']:
            esg_indices[category] = len(labels)
            labels.append(category)
        
        # Add top concepts
        top_concepts = sorted(self.concepts, key=lambda x: x['confidence'], reverse=True)[:20]
        concept_indices = {}
        for concept in top_concepts:
            concept_indices[concept['text']] = len(labels)
            # Truncate long concept names
            display_name = concept['text'][:30] + "..." if len(concept['text']) > 30 else concept['text']
            labels.append(display_name)
        
        # Add evidence pages
        page_indices = {}
        for i in range(1, 11):  # Top 10 pages
            page_indices[f"Page {i}"] = len(labels)
            labels.append(f"Page {i}")
        
        # Create flows
        for concept in top_concepts:
            principle = self.map_concept_to_principle(concept['text'])
            esg_category = self.categorize_concept(concept['text'])
            
            if esg_category != 'Other':
                # Principle → ESG
                sources.append(principle_indices[principle])
                targets.append(esg_indices[esg_category])
                values.append(1)
                
                # ESG → Concept
                sources.append(esg_indices[esg_category])
                targets.append(concept_indices[concept['text']])
                values.append(1)
                
                # Concept → Page
                if concept['page_number'] <= 10:
                    page_key = f"Page {concept['page_number']}"
                    sources.append(concept_indices[concept['text']])
                    targets.append(page_indices[page_key])
                    values.append(1)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=15,
                line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])
        
        fig.update_layout(
            title="Detailed BRSR Principles to Evidence Flow",
            font=dict(size=10),
            height=1000
        )
        
        fig.write_html(output_path)
        print(f"Detailed Sankey diagram saved to {output_path}")
        
        return fig
    
    def generate_flow_statistics(self):
        """
        Generate statistics about the flows
        """
        stats = {
            'principles': defaultdict(int),
            'esg_categories': defaultdict(int),
            'total_evidence': 0
        }
        
        for concept in self.concepts:
            if concept['confidence'] >= 0.7:
                principle = self.map_concept_to_principle(concept['text'])
                esg_category = self.categorize_concept(concept['text'])
                
                stats['principles'][principle] += 1
                stats['esg_categories'][esg_category] += 1
                stats['total_evidence'] += 1
        
        print("\n" + "="*80)
        print("BRSR PRINCIPLES TO EVIDENCE FLOW STATISTICS")
        print("="*80)
        
        print("\nConcepts by BRSR Principle:")
        for code in sorted(stats['principles'].keys()):
            count = stats['principles'][code]
            percentage = (count / stats['total_evidence']) * 100
            print(f"  {code} ({self.brsr_principles[code]}): {count} ({percentage:.1f}%)")
        
        print("\nConcepts by ESG Category:")
        for category in sorted(stats['esg_categories'].keys()):
            count = stats['esg_categories'][category]
            percentage = (count / stats['total_evidence']) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print(f"\nTotal High-Confidence Evidence: {stats['total_evidence']}")
        print("="*80 + "\n")
        
        return stats


def main():
    """Main function to create BRSR Sankey diagrams"""
    
    print("Creating BRSR Principles to Evidence Sankey Diagrams...")
    
    # Initialize visualizer
    visualizer = BRSRSankeyVisualizer("d:/ArcTechnologies/concept_map.json")
    
    # Generate statistics
    stats = visualizer.generate_flow_statistics()
    
    # Create main Sankey diagram
    print("\nGenerating main Sankey diagram...")
    visualizer.create_principle_to_evidence_sankey("d:/ArcTechnologies/brsr_sankey.html")
    
    # Create detailed Sankey diagram
    print("\nGenerating detailed Sankey diagram...")
    visualizer.create_detailed_sankey("d:/ArcTechnologies/brsr_detailed_sankey.html")
    
    print("\n" + "="*80)
    print("SANKEY DIAGRAMS CREATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  - brsr_sankey.html: Main BRSR Principles → ESG → Evidence flow")
    print("  - brsr_detailed_sankey.html: Detailed flow with top concepts")
    print("\nOpen these HTML files in a web browser to view interactive diagrams.")


if __name__ == "__main__":
    main()
