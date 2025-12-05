"""
Color-Coded Drift Dashboard
Visualizes concept drift, confidence changes, and relationship evolution across the document
"""

import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict
import pandas as pd


class DriftDashboard:
    """
    Creates an interactive dashboard showing concept and relationship drift
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
        
        # Organize data by pages
        self.organize_by_pages()
        
    def organize_by_pages(self):
        """Organize concepts and relations by page numbers"""
        self.concepts_by_page = defaultdict(list)
        self.relations_by_page = defaultdict(list)
        
        for concept in self.concepts:
            page = concept['page_number']
            self.concepts_by_page[page].append(concept)
        
        for relation in self.relations:
            page = relation['page_number']
            self.relations_by_page[page].append(relation)
    
    def calculate_drift_metrics(self):
        """Calculate drift metrics across pages"""
        pages = sorted(self.concepts_by_page.keys())
        
        metrics = {
            'pages': [],
            'concept_count': [],
            'avg_confidence': [],
            'relation_count': [],
            'new_concepts': [],
            'confidence_drift': [],
            'relation_density': []
        }
        
        seen_concepts = set()
        prev_avg_confidence = None
        
        for page in pages:
            concepts = self.concepts_by_page[page]
            relations = self.relations_by_page[page]
            
            # Concept count
            concept_count = len(concepts)
            
            # Average confidence
            if concepts:
                avg_conf = np.mean([c['confidence'] for c in concepts])
            else:
                avg_conf = 0
            
            # New concepts (not seen before)
            new_count = 0
            for concept in concepts:
                if concept['text'] not in seen_concepts:
                    new_count += 1
                    seen_concepts.add(concept['text'])
            
            # Confidence drift (change from previous page)
            if prev_avg_confidence is not None:
                conf_drift = avg_conf - prev_avg_confidence
            else:
                conf_drift = 0
            
            # Relation density (relations per concept)
            if concept_count > 0:
                rel_density = len(relations) / concept_count
            else:
                rel_density = 0
            
            metrics['pages'].append(page)
            metrics['concept_count'].append(concept_count)
            metrics['avg_confidence'].append(avg_conf)
            metrics['relation_count'].append(len(relations))
            metrics['new_concepts'].append(new_count)
            metrics['confidence_drift'].append(conf_drift)
            metrics['relation_density'].append(rel_density)
            
            prev_avg_confidence = avg_conf
        
        return metrics
    
    def get_drift_color(self, value, threshold=0.05):
        """
        Get color based on drift value
        
        Args:
            value: Drift value
            threshold: Threshold for significant drift
            
        Returns:
            Color string
        """
        if abs(value) < threshold:
            return 'rgba(144, 238, 144, 0.7)'  # Light green - stable
        elif value > threshold:
            return 'rgba(135, 206, 250, 0.7)'  # Light blue - positive drift
        else:
            return 'rgba(255, 182, 193, 0.7)'  # Light red - negative drift
    
    def create_drift_dashboard(self, output_path: str = "drift_dashboard.html"):
        """
        Create comprehensive drift dashboard
        
        Args:
            output_path: Path to save the HTML file
        """
        metrics = self.calculate_drift_metrics()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Concept Count by Page',
                'Average Confidence by Page',
                'Confidence Drift (Color-Coded)',
                'New Concepts Discovery',
                'Relationship Density',
                'Drift Heatmap'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'heatmap'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # 1. Concept Count by Page
        fig.add_trace(
            go.Scatter(
                x=metrics['pages'],
                y=metrics['concept_count'],
                mode='lines+markers',
                name='Concept Count',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ),
            row=1, col=1
        )
        
        # 2. Average Confidence by Page
        fig.add_trace(
            go.Scatter(
                x=metrics['pages'],
                y=metrics['avg_confidence'],
                mode='lines+markers',
                name='Avg Confidence',
                line=dict(color='#2ecc71', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.2)'
            ),
            row=1, col=2
        )
        
        # Add confidence threshold line
        fig.add_hline(
            y=0.7, line_dash="dash", line_color="red",
            annotation_text="High Confidence Threshold",
            row=1, col=2
        )
        
        # 3. Confidence Drift (Color-Coded)
        colors = [self.get_drift_color(d) for d in metrics['confidence_drift']]
        
        fig.add_trace(
            go.Bar(
                x=metrics['pages'],
                y=metrics['confidence_drift'],
                name='Confidence Drift',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(0,0,0,0.3)', width=1)
                ),
                text=[f"{d:+.3f}" for d in metrics['confidence_drift']],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=1)
        
        # 4. New Concepts Discovery
        fig.add_trace(
            go.Scatter(
                x=metrics['pages'],
                y=metrics['new_concepts'],
                mode='lines+markers',
                name='New Concepts',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=10, symbol='diamond'),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)'
            ),
            row=2, col=2
        )
        
        # 5. Relationship Density
        fig.add_trace(
            go.Scatter(
                x=metrics['pages'],
                y=metrics['relation_density'],
                mode='lines+markers',
                name='Relation Density',
                line=dict(color='#9b59b6', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(155, 89, 182, 0.2)'
            ),
            row=3, col=1
        )
        
        # 6. Drift Heatmap
        # Create a matrix of drift metrics
        heatmap_data = np.array([
            metrics['confidence_drift'],
            [c / max(metrics['concept_count']) if max(metrics['concept_count']) > 0 else 0 
             for c in metrics['concept_count']],
            [n / max(metrics['new_concepts']) if max(metrics['new_concepts']) > 0 else 0 
             for n in metrics['new_concepts']],
            [r / max(metrics['relation_density']) if max(metrics['relation_density']) > 0 else 0 
             for r in metrics['relation_density']]
        ])
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=metrics['pages'],
                y=['Confidence Drift', 'Concept Count', 'New Concepts', 'Relation Density'],
                colorscale='RdYlGn',
                zmid=0,
                text=heatmap_data,
                texttemplate='%{text:.2f}',
                textfont={"size": 8},
                colorbar=dict(title="Normalized<br>Value")
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Concept Drift Dashboard<br>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 28, 'color': '#2c3e50', 'family': 'Arial Black'}
            },
            showlegend=False,
            height=1200,
            plot_bgcolor='rgba(248, 249, 250, 1)',
            paper_bgcolor='white',
            font=dict(size=11, family="Arial, sans-serif")
        )
        
        # Update axes
        fig.update_xaxes(title_text="Page Number", row=1, col=1, gridcolor='lightgray')
        fig.update_xaxes(title_text="Page Number", row=1, col=2, gridcolor='lightgray')
        fig.update_xaxes(title_text="Page Number", row=2, col=1, gridcolor='lightgray')
        fig.update_xaxes(title_text="Page Number", row=2, col=2, gridcolor='lightgray')
        fig.update_xaxes(title_text="Page Number", row=3, col=1, gridcolor='lightgray')
        fig.update_xaxes(title_text="Page Number", row=3, col=2, gridcolor='lightgray')
        
        fig.update_yaxes(title_text="Count", row=1, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Confidence", row=1, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Drift", row=2, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Count", row=2, col=2, gridcolor='lightgray')
        fig.update_yaxes(title_text="Density", row=3, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Metric", row=3, col=2, gridcolor='lightgray')
        
        # Save to HTML
        fig.write_html(output_path)
        print(f"Drift dashboard saved to {output_path}")
        
        return fig, metrics
    
    def create_category_drift_dashboard(self, output_path: str = "category_drift_dashboard.html"):
        """
        Create category-specific drift dashboard (ESG categories)
        
        Args:
            output_path: Path to save the HTML file
        """
        # Categorize concepts
        categories = {
            'Environmental': ['environment', 'climate', 'carbon', 'emission', 'energy', 'renewable'],
            'Social': ['employee', 'social', 'community', 'diversity', 'health', 'safety'],
            'Governance': ['governance', 'ethics', 'compliance', 'transparency', 'board']
        }
        
        # Track category counts by page
        pages = sorted(self.concepts_by_page.keys())
        category_data = {cat: [] for cat in categories}
        
        for page in pages:
            concepts = self.concepts_by_page[page]
            counts = {cat: 0 for cat in categories}
            
            for concept in concepts:
                text_lower = concept['text'].lower()
                for cat, keywords in categories.items():
                    if any(kw in text_lower for kw in keywords):
                        counts[cat] += 1
                        break
            
            for cat in categories:
                category_data[cat].append(counts[cat])
        
        # Create stacked area chart
        fig = go.Figure()
        
        colors = {
            'Environmental': 'rgba(46, 204, 113, 0.7)',
            'Social': 'rgba(52, 152, 219, 0.7)',
            'Governance': 'rgba(231, 76, 60, 0.7)'
        }
        
        for cat in categories:
            fig.add_trace(go.Scatter(
                x=pages,
                y=category_data[cat],
                mode='lines',
                name=cat,
                stackgroup='one',
                fillcolor=colors[cat],
                line=dict(width=0.5, color=colors[cat].replace('0.7', '1'))
            ))
        
        fig.update_layout(
            title={
                'text': "ESG Category Drift Dashboard<br><sub>Environmental, Social, and Governance concept evolution</sub>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#2c3e50'}
            },
            xaxis_title="Page Number",
            yaxis_title="Concept Count",
            hovermode='x unified',
            height=600,
            plot_bgcolor='rgba(248, 249, 250, 1)',
            paper_bgcolor='white',
            font=dict(size=12, family="Arial, sans-serif")
        )
        
        fig.write_html(output_path)
        print(f"Category drift dashboard saved to {output_path}")
        
        return fig
    
    def generate_drift_report(self, output_path: str = "drift_report.txt"):
        """
        Generate a text report of drift analysis
        
        Args:
            output_path: Path to save the report
        """
        metrics = self.calculate_drift_metrics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CONCEPT DRIFT ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Pages Analyzed: {len(metrics['pages'])}\n")
            f.write(f"Total Concepts: {sum(metrics['concept_count'])}\n")
            f.write(f"Average Concepts per Page: {np.mean(metrics['concept_count']):.2f}\n")
            f.write(f"Average Confidence: {np.mean(metrics['avg_confidence']):.3f}\n")
            f.write(f"Total Relationships: {sum(metrics['relation_count'])}\n\n")
            
            # Drift analysis
            f.write("DRIFT ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            # Find pages with significant drift
            significant_drift = []
            for i, drift in enumerate(metrics['confidence_drift']):
                if abs(drift) > 0.1:
                    page = metrics['pages'][i]
                    significant_drift.append((page, drift))
            
            if significant_drift:
                f.write("Pages with Significant Confidence Drift (>0.1):\n")
                for page, drift in significant_drift:
                    direction = "↑ Increase" if drift > 0 else "↓ Decrease"
                    f.write(f"  Page {page}: {drift:+.3f} ({direction})\n")
            else:
                f.write("No significant confidence drift detected.\n")
            
            f.write("\n")
            
            # Peak discovery pages
            f.write("CONCEPT DISCOVERY PEAKS\n")
            f.write("-"*80 + "\n")
            top_discovery = sorted(zip(metrics['pages'], metrics['new_concepts']), 
                                  key=lambda x: x[1], reverse=True)[:5]
            f.write("Top 5 Pages with Most New Concepts:\n")
            for page, count in top_discovery:
                f.write(f"  Page {page}: {count} new concepts\n")
            
            f.write("\n")
            
            # Relationship density
            f.write("RELATIONSHIP DENSITY ANALYSIS\n")
            f.write("-"*80 + "\n")
            avg_density = np.mean(metrics['relation_density'])
            f.write(f"Average Relationship Density: {avg_density:.2f} relations per concept\n")
            
            high_density = [(p, d) for p, d in zip(metrics['pages'], metrics['relation_density']) 
                           if d > avg_density * 1.5]
            if high_density:
                f.write("\nPages with High Relationship Density (>1.5x average):\n")
                for page, density in high_density:
                    f.write(f"  Page {page}: {density:.2f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"Drift report saved to {output_path}")


def main():
    """Main function to create drift dashboards"""
    
    print("Creating Color-Coded Drift Dashboards...")
    print("="*80)
    
    # Initialize dashboard
    dashboard = DriftDashboard("d:/ArcTechnologies/concept_map.json")
    
    # Create main drift dashboard
    print("\n1. Generating main drift dashboard...")
    fig, metrics = dashboard.create_drift_dashboard("d:/ArcTechnologies/drift_dashboard.html")
    
    # Create category drift dashboard
    print("\n2. Generating ESG category drift dashboard...")
    dashboard.create_category_drift_dashboard("d:/ArcTechnologies/category_drift_dashboard.html")
    
    # Generate drift report
    print("\n3. Generating drift analysis report...")
    dashboard.generate_drift_report("d:/ArcTechnologies/drift_report.txt")
    
    # Print summary
    print("\n" + "="*80)
    print("DRIFT DASHBOARD SUMMARY")
    print("="*80)
    print(f"Pages Analyzed: {len(metrics['pages'])}")
    print(f"Total Concepts: {sum(metrics['concept_count'])}")
    print(f"Average Confidence: {np.mean(metrics['avg_confidence']):.3f}")
    print(f"Max Confidence Drift: {max(metrics['confidence_drift'], key=abs):+.3f}")
    print(f"Average Relation Density: {np.mean(metrics['relation_density']):.2f}")
    
    print("\n" + "="*80)
    print("DRIFT DASHBOARDS CREATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  - drift_dashboard.html: Main drift dashboard with 6 visualizations")
    print("  - category_drift_dashboard.html: ESG category evolution")
    print("  - drift_report.txt: Detailed drift analysis report")
    print("\nOpen the HTML files in a web browser to view interactive dashboards.")
    print("\nColor Legend:")
    print("  🟢 Green: Stable (drift < 0.05)")
    print("  🔵 Blue: Positive drift (increasing confidence)")
    print("  🔴 Red: Negative drift (decreasing confidence)")


if __name__ == "__main__":
    main()
