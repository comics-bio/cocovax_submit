import json
from cmd import PROMPT
from pdfminer.high_level import extract_text

knowledge = """
Codon pair bias (CPB) is the nonrandom occurrence of specific codon pairs in a genome, deviating from statistical expectations. This bias affects mRNA stability, protein translation efficiency, and host immune responses. In contrast, codon pair deoptimization (CPD) increases the frequency of underrepresented—or “suboptimal”—codon pairs (often those enriched in unfavorable dinucleotides such as CpG or UpA), which leads to reduced viral protein synthesis, lower replication rates, and ultimately attenuated virulence.
Exploiting CPD has emerged as a promising strategy for viral attenuation. By strategically reorganizing synonymous codons within the viral genome, CPD elevates the occurrence of rare codon pairs without altering the encoded proteins. This recoding reduces the pathogen’s replicative fitness while preserving its antigenic properties, enabling the attenuated pathogen to effectively stimulate protective immune responses without causing disease.
The Codon Adaptation Index (CAI) quantifies how closely a gene’s codon usage matches the host’s preferred codons. Higher CAI values generally correlate with increased translation efficiency. In CAI deoptimization, host-rare codons are deliberately incorporated into viral genes to reduce protein production and viral fitness. Like CPD, CAI deoptimization attenuates virulence while maintaining the pathogen’s immunogenic characteristics.
"""
prompt = """
Please fully read the literature provided below carefully and generate a JSON object of this literature with the following structure. Make sure you have read the literature carefully and fully understand the content before generating the JSON object. Don't omit any important information from the literature. The JSON object should contain the following fields:
{
  "type": ["<one or more from 'Codon Pair Deoptimization (CPD)', 'Codon Adaptation Index (CAI) Deoptimization'>"],
  "use codon pair deoptimization (CPD)": "<one of 'true', 'false'>",
  "use codon adaption index (CAI) deoptimization": "<one of 'true', 'false'>",
  "has animal experiment": "<one of 'true', 'false'>",
  "animal experiment content": "<brief text>",
  "model animals": ["<model animals>",],
  "experiment results": "<brief text>",
  "experiments in vitro": "<brief text>",
  "experiments in vivo": "<brief text>",
  "live attenuation strategy": "<brief text>",
  "deoptimized gene target": ["<gene names>",],
  "main research content": "<brief text>",
  "vaccine advantages": ["<some key words>"],
  "targeted pathogen": ["<pathogen names>",],
  "host of targeted pathogen": ["<host names>",],
  "targeted pathogen strain": ["<pathogen strains>",],
  "score": "<score>",
  "score reasons": "<brief text>",
  "research novelty": "<brief text>"
}
- The "type" field must be an array containing one or more items selected from: 'Codon Pair Deoptimization (CPD)', 'Codon Adaptation Index (CAI) Deoptimization'.
- The "use codon pair deoptimization" field must be one of the following: 'true', 'false'. Codon pair deoptimization (CPD) involves the strategic reorganization of synonymous codons within the viral genome to increase the occurrence of statistically underrepresented (suboptimal) codon pairs.
- The "use codon deoptimization" field must be one of the following: 'true', 'false'. use codon deoptimization is the same as CAI deoptimization, which involves the strategic incorporation of host-rare codons into viral genes.
- The "has animal experiment" field must be one of the following: 'true', 'false'.
- The "animal experiment content" field should contain 1-2 sentences describing the animal experiment.
- The "alive attenuation strategy" field should describing the live attenuation strategy.
- The "research content" field should contain 1-2 sentences describing the research content.
- The "vaccine advantages" field should be an array of key words that describe the advantages of the vaccine designed in the literature.
- The "pathogen" field should be an array containing the name of the pathogen used for live attenuation in the literature. Multiple choice from ['Chikungunya Virus', 'Classical Swine Fever Virus', 'Crimean-Congo Hemorrhagic Fever Virus', 'Dengue Virus', 'Enterovirus A71', 'Foot-and-Mouth Disease Virus', 'Herpesvirus', 'Infectious Laryngotracheitis Virus', 'Influenza Virus', 'Lassa Virus', 'Lymphocytic Choriomeningitis Virus', 'Macrobrachium Rosenbergii Nodavirus', "Marek's Disease Virus", 'Melon Necrotic Spot Virus', 'Newcastle Disease Virus', 'Parainfluenza Virus', 'Poliovirus', 'Porcine Reproductive and Respiratory Syndrome Virus', 'Rabies Virus', 'Respiratory Syncytial Virus', 'SARS-CoV-2', 'Salmonella Typhimurium', 'Simian Immunodeficiency Virus', 'Streptococcus pneumoniae', 'Tick-Borne Encephalitis Virus', 'Vesicular Stomatitis Virus', 'Zika Virus']
- The "score" field should be a number between 0 and 10, indicating the new designed vaccine's score based on the literature. The higher the score, the better the vaccine design. The reasons for the score should be included in the "score reasons" field.
- The "score reasons" field should contain 1-2 sentences explaining the reasons for the score.
Please make sure the JSON object is well-structured and contains all the necessary information from the literature.
Literature Contents:
"""


def generate_json(id, text):
    dic = {"custom_id": id,
           "method": "POST",
           "url": "/v1/chat/completions",
           "body": {"model": "qwq-plus",
                    "messages": [{"role": "user",
                                  "content": knowledge + '\n' + prompt + text}]}, }
    return json.dumps(dic, separators=(',', ':'), ensure_ascii=False)


jsonl = ''
for file in os.listdir('pdf'):
    if not file.endswith('.pdf'):
        continue
    print(file)
    text = extract_text(f'./pdf/{file}')
    filtered_lines = [line for line in text.splitlines() if len(line) >= 3]
    filtered_text = "\n".join(filtered_lines)
    if len(filtered_text.split()) > 50000:
        filtered_text = ' '.join(filtered_text.split()[:50000])
    jsonl += generate_json(file, filtered_text) + '\n'


with open('jsonl.jsonl', 'w', encoding='utf-8') as f:
    f.write(jsonl)
