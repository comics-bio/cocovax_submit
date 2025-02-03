import json
from pdfminer.high_level import extract_text
knowledge = """
Codon pair bias (CPB) is a genetic phenomenon where specific codon pairs occur at frequencies that deviate from statistical expectations within a genome. This bias significantly influences protein translation efficiency, mRNA stability, and host immune responses to viral infections. Codon pair deoptimization, which involves increasing the frequency of rare codon pairs or those containing unfavorable dinucleotides (such as CpG or UpA), can effectively reduce viral replication rates and attenuate viral virulence.
The exploitation of codon pair bias deoptimization has emerged as a promising strategy for viral attenuation. Codon pair deoptimization (CPD) involves the strategic reorganization of synonymous codons within the viral genome to increase the occurrence of statistically underrepresented (suboptimal) codon pairs. This recoding approach reduces viral protein expression and replication efficiency, thereby attenuating viral virulence while preserving antigenic properties. The resulting weakened virus can effectively stimulate immune responses without causing disease manifestation.
The Codon Adaptation Index (CAI) quantifies the relative optimization of codon usage within a gene or species by measuring the bias toward specific synonymous codons. This index evaluates the alignment between viral codon usage and host-preferred codons, where higher CAI values indicate more efficient and optimal codon usage for the host organism. This metric provides valuable insights into the translational efficiency of viral proteins within host cells.
CAI deoptimization involves the strategic incorporation of host-rare codons into viral genes. This modification reduces viral fitness in natural hosts by decreasing viral protein expression and replication efficiency. Like CPD, this approach attenuates viral virulence while maintaining the virus's antigenic characteristics.
"""
prompt = """
Please detailed read the literature provided below carefully and generate a JSON object of this literature with the following structure:
{
  "type": ["<one or more from 'Codon Pair Deoptimization (CPD)', 'Codon Adaptation Index (CAI) Deoptimization'>"],
  "use codon pair deoptimization": "<one of 'true', 'false'>",
  "use codon adaptation index deoptimization": "<one of 'true', 'false'>",
  "has animal experiment": "<one of 'true', 'false'>",
  "animal experiment content": "<brief text>",
  "live attenuation strategy": "<brief text>",
  "research content": "<brief text>",
  "vaccine advantages": ["<some key words>"],
  "virus": ["<virus name>"],
  "score": "<score>",
  "score reasons": "<brief text>",
}
- The "type" field must be an array containing one or more items selected from: 'Codon Pair Deoptimization (CPD)', 'Codon Adaptation Index (CAI) Deoptimization'.
- The "use codon pair deoptimization" field must be one of the following: 'true', 'false'.
- The "use codon adaptation index deoptimization" field must be one of the following: 'true', 'false'.
- The "has animal experiment" field must be one of the following: 'true', 'false'.
- The "animal experiment content" field should contain 1-2 sentences describing the animal experiment.
- The "alive attenuation strategy" field should describing the live attenuation strategy.
- The "research content" field should contain 1-2 sentences describing the research content.
- The "vaccine advantages" field should be an array of key words that describe the advantages of the vaccine designed in the literature.
- The "virus" field should be an array containing the name of the virus used for live attenuation in the literature.
- The "score" field should be a number between 0 and 10, indicating the new designed vaccine's score based on the literature. The higher the score, the better the vaccine design. The reasons for the score should be included in the "score reasons" field.
- The "score reasons" field should contain 1-2 sentences explaining the reasons for the score.
Please generate the JSON object based on the literature provided below:
Literature Contents:
"""


def generate_json(id, text):
    dic = {"custom_id": id,
           "method": "POST",
           "url": "/v1/chat/completions",
           "body": {"model": "qwen-max",
                     "messages": [{"role": "system",
                                   "content": "You are a helpful assistant with expertise in biology and medicine."},
                                  {"role": "user",
                                   "content": "knowledge"},
                                  {"role": "user",
                                   "content": prompt + text}]}, }
    return json.dumps(dic, separators=(',', ':'), ensure_ascii=False)
