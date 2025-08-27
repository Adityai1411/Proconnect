import spacy
from sentence_transformers import SentenceTransformer, util
import yake

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_features(text: str):
    doc = nlp(text)
    keywords = [kw for kw, _ in yake.KeywordExtractor(top=15).extract_keywords(text)]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    nouns = [t.text for t in doc if t.pos_ in ['NOUN','PROPN'] and not t.is_stop]

    sentences = [s.text for s in doc.sents]
    platforms = {
        "LinkedIn": "Formal job experience, education, and skills summary",
        "Instagram": "Achievements, certifications, and short highlights",
        "GitHub": "Technical projects, tools used, and code links",
        "Google Sites": "Full bio, skills, education, and all project info"
    }
    platform_embeddings = {k: util.normalize_embeddings(model.encode([v], convert_to_tensor=True)).squeeze(0) for k,v in platforms.items()}
    sentence_embeddings = util.normalize_embeddings(model.encode(sentences, convert_to_tensor=True))

    sentence_to_platform = {}
    for i, s in enumerate(sentences):
        best_match = max(platforms.keys(),
                         key=lambda k: util.pytorch_cos_sim(sentence_embeddings[i], platform_embeddings[k]).item())
        sentence_to_platform[s] = best_match

    return {
        "keywords": keywords,
        "entities": entities,
        "nouns": nouns,
        "sentence_to_platform": sentence_to_platform,
        "score": 0
    }
