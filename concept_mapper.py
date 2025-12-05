"""
Faithful Concept Mapper - GenAI Intern Evaluation
This module extracts concepts from PDF documents and maps their relationships
while ensuring faithfulness to the source material.
"""

import os
import re
import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict


@dataclass
class Concept:
    """Represents a concept extracted from the document"""
    text: str
    context: str
    page_number: int
    sentence: str
    confidence: float
    
    def to_dict(self):
        data = asdict(self)
        data['confidence'] = float(data['confidence'])
        return data


@dataclass
class ConceptRelation:
    """Represents a relationship between two concepts"""
    source: str
    target: str
    relation_type: str
    evidence: str
    page_number: int
    confidence: float
    
    def to_dict(self):
        data = asdict(self)
        data['confidence'] = float(data['confidence'])
        return data


class FaithfulConceptMapper:
    """
    Extracts concepts from documents and maps their relationships
    while ensuring faithfulness to the source material.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Concept Mapper
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(model_name)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spacy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.concepts: List[Concept] = []
        self.relations: List[ConceptRelation] = []
        self.concept_embeddings = None
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """
        Extract text from PDF file page by page
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to text content
        """
        page_texts = {}
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                page_texts[page_num + 1] = text
                
        return page_texts
    
    def extract_concepts(self, page_texts: Dict[int, str], 
                        min_confidence: float = 0.5) -> List[Concept]:
        """
        Extract concepts from text using NLP techniques
        
        Args:
            page_texts: Dictionary mapping page numbers to text
            min_confidence: Minimum confidence threshold for concept extraction
            
        Returns:
            List of extracted concepts
        """
        concepts = []
        seen_concepts = set()
        
        for page_num, text in page_texts.items():
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract sentences
            sentences = [sent.text.strip() for sent in doc.sents]
            
            for sentence in sentences:
                sent_doc = self.nlp(sentence)
                
                # Extract noun phrases as potential concepts
                for chunk in sent_doc.noun_chunks:
                    concept_text = chunk.text.strip().lower()
                    
                    # Filter out very short or very long concepts
                    if len(concept_text.split()) < 2 or len(concept_text.split()) > 5:
                        continue
                    
                    # Skip if already seen
                    if concept_text in seen_concepts:
                        continue
                    
                    # Calculate confidence based on various factors
                    confidence = self._calculate_concept_confidence(chunk, sent_doc)
                    
                    if confidence >= min_confidence:
                        # Get context (surrounding text)
                        context = self._get_context(sentence, concept_text)
                        
                        concept = Concept(
                            text=concept_text,
                            context=context,
                            page_number=page_num,
                            sentence=sentence,
                            confidence=confidence
                        )
                        
                        concepts.append(concept)
                        seen_concepts.add(concept_text)
        
        self.concepts = concepts
        return concepts
    
    def _calculate_concept_confidence(self, chunk, doc) -> float:
        """
        Calculate confidence score for a concept
        
        Args:
            chunk: spaCy noun chunk
            doc: spaCy document
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence
        
        # Increase confidence for proper nouns
        if any(token.pos_ == "PROPN" for token in chunk):
            confidence += 0.2
        
        # Increase confidence for capitalized terms
        if chunk.text[0].isupper():
            confidence += 0.1
        
        # Increase confidence for domain-specific terms
        domain_keywords = ['sustainability', 'environmental', 'social', 'governance', 
                          'carbon', 'emission', 'renewable', 'energy', 'waste',
                          'responsibility', 'stakeholder', 'community', 'employee']
        
        if any(keyword in chunk.text.lower() for keyword in domain_keywords):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _get_context(self, sentence: str, concept: str, window: int = 50) -> str:
        """
        Get context around a concept in a sentence
        
        Args:
            sentence: The sentence containing the concept
            concept: The concept text
            window: Number of characters to include on each side
            
        Returns:
            Context string
        """
        concept_pos = sentence.lower().find(concept.lower())
        if concept_pos == -1:
            return sentence[:200]
        
        start = max(0, concept_pos - window)
        end = min(len(sentence), concept_pos + len(concept) + window)
        
        return sentence[start:end]
    
    def map_relationships(self, similarity_threshold: float = 0.5) -> List[ConceptRelation]:
        """
        Map relationships between concepts using semantic similarity and co-occurrence
        
        Args:
            similarity_threshold: Minimum similarity score to create a relationship
            
        Returns:
            List of concept relationships
        """
        if not self.concepts:
            raise ValueError("No concepts extracted. Run extract_concepts first.")
        
        # Create embeddings for all concepts
        concept_texts = [c.text for c in self.concepts]
        self.concept_embeddings = self.embedding_model.encode(concept_texts)
        
        relations = []
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(self.concept_embeddings)
        
        for i, concept1 in enumerate(self.concepts):
            for j, concept2 in enumerate(self.concepts):
                if i >= j:  # Avoid duplicates and self-relations
                    continue
                
                similarity = similarities[i][j]
                
                if similarity >= similarity_threshold:
                    # Determine relation type based on context
                    relation_type = self._determine_relation_type(
                        concept1, concept2, similarity
                    )
                    
                    # Find evidence for the relationship
                    evidence = self._find_relationship_evidence(concept1, concept2)
                    
                    if evidence:
                        relation = ConceptRelation(
                            source=concept1.text,
                            target=concept2.text,
                            relation_type=relation_type,
                            evidence=evidence,
                            page_number=concept1.page_number,
                            confidence=similarity
                        )
                        
                        relations.append(relation)
        
        self.relations = relations
        return relations
    
    def _determine_relation_type(self, concept1: Concept, concept2: Concept, 
                                 similarity: float) -> str:
        """
        Determine the type of relationship between two concepts
        
        Args:
            concept1: First concept
            concept2: Second concept
            similarity: Similarity score
            
        Returns:
            Relationship type as string
        """
        # Check for hierarchical relationships
        if concept2.text in concept1.text or concept1.text in concept2.text:
            return "part_of"
        
        # Check for causal relationships
        causal_indicators = ['leads to', 'results in', 'causes', 'impacts', 'affects']
        combined_context = concept1.sentence + " " + concept2.sentence
        
        if any(indicator in combined_context.lower() for indicator in causal_indicators):
            return "causal"
        
        # Check for temporal relationships
        temporal_indicators = ['before', 'after', 'during', 'following', 'prior to']
        if any(indicator in combined_context.lower() for indicator in temporal_indicators):
            return "temporal"
        
        # Default to semantic similarity
        if similarity > 0.7:
            return "strongly_related"
        else:
            return "related"
    
    def _find_relationship_evidence(self, concept1: Concept, concept2: Concept) -> str:
        """
        Find textual evidence for the relationship between two concepts
        
        Args:
            concept1: First concept
            concept2: Second concept
            
        Returns:
            Evidence text or empty string
        """
        # Check if concepts appear in the same sentence
        if concept1.sentence == concept2.sentence:
            return concept1.sentence
        
        # Check if concepts appear on the same page
        if concept1.page_number == concept2.page_number:
            return f"Both concepts appear on page {concept1.page_number}"
        
        # Check context overlap
        context1_words = set(concept1.context.lower().split())
        context2_words = set(concept2.context.lower().split())
        overlap = context1_words.intersection(context2_words)
        
        if len(overlap) > 3:
            return f"Concepts share contextual terms: {', '.join(list(overlap)[:5])}"
        
        return ""
    
    def visualize_concept_map(self, output_path: str = "concept_map.png", 
                             max_concepts: int = 50):
        """
        Visualize the concept map as a network graph
        
        Args:
            output_path: Path to save the visualization
            max_concepts: Maximum number of concepts to visualize
        """
        if not self.concepts or not self.relations:
            raise ValueError("No concepts or relations to visualize")
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes (concepts) - limit to top concepts by confidence
        top_concepts = sorted(self.concepts, key=lambda x: x.confidence, reverse=True)[:max_concepts]
        concept_set = {c.text for c in top_concepts}
        
        for concept in top_concepts:
            G.add_node(concept.text, confidence=concept.confidence)
        
        # Add edges (relationships) - only for concepts in the graph
        for relation in self.relations:
            if relation.source in concept_set and relation.target in concept_set:
                G.add_edge(
                    relation.source, 
                    relation.target,
                    relation_type=relation.relation_type,
                    confidence=relation.confidence
                )
        
        # Create visualization
        plt.figure(figsize=(20, 16))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes with size based on confidence
        node_sizes = [G.nodes[node]['confidence'] * 1000 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7)
        
        # Draw edges with width based on confidence
        edge_widths = [G.edges[edge]['confidence'] * 3 for edge in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              alpha=0.5, arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title("Faithful Concept Map", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Concept map saved to {output_path}")
    
    def export_to_json(self, output_path: str = "concept_map.json"):
        """
        Export concepts and relationships to JSON
        
        Args:
            output_path: Path to save the JSON file
        """
        data = {
            "concepts": [c.to_dict() for c in self.concepts],
            "relations": [r.to_dict() for r in self.relations],
            "metadata": {
                "total_concepts": len(self.concepts),
                "total_relations": len(self.relations)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Concept map exported to {output_path}")
    
    def generate_report(self, output_path: str = "concept_map_report.txt"):
        """
        Generate a human-readable report of the concept map
        
        Args:
            output_path: Path to save the report
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FAITHFUL CONCEPT MAP REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Concepts Extracted: {len(self.concepts)}\n")
            f.write(f"Total Relationships Mapped: {len(self.relations)}\n")
            f.write(f"Average Concept Confidence: {np.mean([c.confidence for c in self.concepts]):.3f}\n")
            f.write(f"Average Relationship Confidence: {np.mean([r.confidence for r in self.relations]):.3f}\n\n")
            
            # Top concepts
            f.write("TOP 20 CONCEPTS (by confidence)\n")
            f.write("-" * 80 + "\n")
            top_concepts = sorted(self.concepts, key=lambda x: x.confidence, reverse=True)[:20]
            for i, concept in enumerate(top_concepts, 1):
                f.write(f"{i}. {concept.text.upper()}\n")
                f.write(f"   Confidence: {concept.confidence:.3f}\n")
                f.write(f"   Page: {concept.page_number}\n")
                f.write(f"   Context: {concept.context[:100]}...\n\n")
            
            # Relationship types distribution
            f.write("\nRELATIONSHIP TYPES DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            relation_types = defaultdict(int)
            for relation in self.relations:
                relation_types[relation.relation_type] += 1
            
            for rel_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{rel_type}: {count}\n")
            
            # Sample relationships
            f.write("\n\nSAMPLE RELATIONSHIPS (Top 10 by confidence)\n")
            f.write("-" * 80 + "\n")
            top_relations = sorted(self.relations, key=lambda x: x.confidence, reverse=True)[:10]
            for i, relation in enumerate(top_relations, 1):
                f.write(f"{i}. {relation.source.upper()} --[{relation.relation_type}]--> {relation.target.upper()}\n")
                f.write(f"   Confidence: {relation.confidence:.3f}\n")
                f.write(f"   Evidence: {relation.evidence[:150]}...\n\n")
        
        print(f"Report generated: {output_path}")


def main():
    """Main function to run the Faithful Concept Mapper"""
    
    # Initialize the mapper
    print("Initializing Faithful Concept Mapper...")
    mapper = FaithfulConceptMapper()
    
    # Path to the sustainability report
    pdf_path = "d:/ArcTechnologies/business-responsibility-and-sustainability-report.pdf"
    
    # Extract text from PDF
    print(f"\nExtracting text from {pdf_path}...")
    page_texts = mapper.extract_text_from_pdf(pdf_path)
    print(f"Extracted text from {len(page_texts)} pages")
    
    # Extract concepts
    print("\nExtracting concepts...")
    concepts = mapper.extract_concepts(page_texts, min_confidence=0.6)
    print(f"Extracted {len(concepts)} concepts")
    
    # Map relationships
    print("\nMapping relationships...")
    relations = mapper.map_relationships(similarity_threshold=0.5)
    print(f"Mapped {len(relations)} relationships")
    
    # Generate outputs
    print("\nGenerating outputs...")
    mapper.visualize_concept_map("d:/ArcTechnologies/concept_map.png", max_concepts=50)
    mapper.export_to_json("d:/ArcTechnologies/concept_map.json")
    mapper.generate_report("d:/ArcTechnologies/concept_map_report.txt")
    
    print("\n" + "=" * 80)
    print("FAITHFUL CONCEPT MAPPER COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - concept_map.png: Visual representation of the concept map")
    print("  - concept_map.json: Machine-readable concept map data")
    print("  - concept_map_report.txt: Human-readable report")


if __name__ == "__main__":
    main()
