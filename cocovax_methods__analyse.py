from matplotlib import pyplot as plt
import math
import random
import string
import subprocess
import re
import heapq
import numpy as np
import json
import edlib
import RNA
import pandas as pd
from Bio import SeqIO
import random
import os

# Codon pair scores dictionary
with open('./cocovax/cps_dict.json', 'r') as f:
    cps_dict = json.load(f)
# Codon Adaptation Index log weights
with open('./cocovax/cai_lnw.json', 'r') as f:
    cai_lnw = json.load(f)
# Codon choices for optimization （most deoptimized codons）
with open('./cocovax/cai_choice.json', 'r') as f:
    cai_choice = json.load(f)

# Define genetic code translation tables
# Maps codons to amino acids
codon_table = {
    'GCT': 'A',
    'GCC': 'A',
    'GCA': 'A',
    'GCG': 'A',
    'CGT': 'R',
    'CGC': 'R',
    'CGA': 'R',
    'CGG': 'R',
    'AGA': 'R',
    'AGG': 'R',
    'TCT': 'S',
    'TCC': 'S',
    'TCA': 'S',
    'TCG': 'S',
    'AGT': 'S',
    'AGC': 'S',
    'ATT': 'I',
    'ATC': 'I',
    'ATA': 'I',
    'TTA': 'L',
    'TTG': 'L',
    'CTT': 'L',
    'CTC': 'L',
    'CTA': 'L',
    'CTG': 'L',
    'GGT': 'G',
    'GGC': 'G',
    'GGA': 'G',
    'GGG': 'G',
    'GTT': 'V',
    'GTC': 'V',
    'GTA': 'V',
    'GTG': 'V',
    'ACT': 'T',
    'ACC': 'T',
    'ACA': 'T',
    'ACG': 'T',
    'CCT': 'P',
    'CCC': 'P',
    'CCA': 'P',
    'CCG': 'P',
    'AAT': 'N',
    'AAC': 'N',
    'GAT': 'D',
    'GAC': 'D',
    'TGT': 'C',
    'TGC': 'C',
    'CAA': 'Q',
    'CAG': 'Q',
    'GAA': 'E',
    'GAG': 'E',
    'CAT': 'H',
    'CAC': 'H',
    'AAA': 'K',
    'AAG': 'K',
    'TTT': 'F',
    'TTC': 'F',
    'TAT': 'Y',
    'TAC': 'Y',
    'ATG': 'M',
    'TGG': 'W',
    'TAG': 'STOP',
    'TGA': 'STOP',
    'TAA': 'STOP'}
# Maps amino acids to possible codons
dict_ammo = {'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
             'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'], 'I': ['ATT', 'ATC', 'ATA'],
             'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], 'G': ['GGT', 'GGC', 'GGA', 'GGG'],
             'V': ['GTT', 'GTC', 'GTA', 'GTG'], 'T': ['ACT', 'ACC', 'ACA', 'ACG'], 'P': ['CCT', 'CCC', 'CCA', 'CCG'],
             'N': ['AAT', 'AAC'], 'D': ['GAT', 'GAC'], 'C': ['TGT', 'TGC'], 'Q': ['CAA', 'CAG'], 'E': ['GAA', 'GAG'],
             'H': ['CAT', 'CAC'], 'K': ['AAA', 'AAG'], 'F': ['TTT', 'TTC'], 'Y': ['TAT', 'TAC'], 'M': ['ATG'],
             'W': ['TGG'], 'STOP': ['TAG', 'TGA', 'TAA']}
# DNA/RNA base pairing
transcription_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}


def get_windows(cds):
    """Calculate window size for RNA analysis"""
    return max(len(cds) // 5, 100)


def rna_fold(cds):
    """Calculate RNA secondary structure and minimum free energy"""
    fc = RNA.fold_compound(cds)
    (ss, mfe) = fc.mfe()
    return ss, mfe


def get_free_energy(cds):
    """Get minimum free energy of RNA structure"""
    return rna_fold(cds)[1]


def get_mutation_distance(s1, s2):
    return edlib.align(s1, s2, mode="NW")['editDistance']


def free_energy_frame(cds, window=None):
    """
    Calculate free energy values for sliding windows across a coding sequence (CDS).

    This function analyzes the local RNA structure stability by calculating
    minimum free energy values in overlapping windows along the sequence.

    Args:
        cds (str): The coding sequence to analyze
        window (int, optional): The size of the sliding window. If None,
                              calculated as max(len(cds)/5, 100)

    Returns:
        tuple: A pair of lists:
            - free_energy (list): List of free energy values for each window
            - x_index (list): List of position indices (as strings) corresponding
                             to the center of each window

    Algorithm:
        1. If no window size provided, calculate it based on sequence length
        2. Slide window across sequence with 50% overlap between windows
        3. For each window:
           - Calculate minimum free energy of the subsequence
           - Store the position index (window center)
    """
    # Set default window size if none provided
    if window is None:
        window = get_windows(cds)

    # Initialize result lists
    free_energy = []  # Store energy values
    x_index = []     # Store position indices

    # Slide window across sequence
    # Step size is half the window size (50% overlap)
    for i in range(window, len(cds) - window // 2 + 1, int(window // 2)):
        # Calculate free energy for current window
        # Window spans from (i-window) to min(i, sequence_length)
        free_energy.append(get_free_energy(cds[i - window:min(i, len(cds))]))

        # Store the center position of current window
        x_index.append(str(int(i - window // 2)))

    return free_energy, x_index


def free_energy_bias(cds1, cds2, window=None):
    """
    Calculate the free energy bias between two coding sequences of equal length.

    This function compares the RNA structure stability patterns between two sequences
    by measuring the absolute differences in their free energy profiles.

    Args:
        cds1 (str): First coding sequence
        cds2 (str): Second coding sequence
        window (int, optional): Size of sliding window for free energy calculation.
                              If None, calculated as max(len(cds1)/5, 100)

    Returns:
        float: Normalized bias value representing the average difference in free energy
               profiles between the two sequences

    Raises:
        ValueError: If the input sequences have different lengths

    Algorithm:
        1. Verify sequences are same length
        2. Calculate free energy profiles for both sequences
        3. Sum absolute differences between corresponding windows
        4. Normalize by effective sequence length (2*length - window)

    Example:
        >>> seq1 = "ATGCATGC"
        >>> seq2 = "ATGCGTGC"
        >>> bias = free_energy_bias(seq1, seq2)
    """
    # Use default window size if none provided
    if window is None:
        window = get_windows(cds1)

    # Validate input sequences have equal length
    if len(cds1) != len(cds2):
        raise ValueError('The length of the two CDSs are not equal')

    # Calculate free energy profiles for both sequences
    # Only keep the energy values ([0]), discard position indices
    free_energy1 = free_energy_frame(cds1, window)[0]
    free_energy2 = free_energy_frame(cds2, window)[0]

    # Calculate cumulative absolute difference between profiles
    bias = 0
    for i in range(len(free_energy1)):
        bias += abs(free_energy1[i] - free_energy2[i])

    # Normalize by effective sequence length
    # (2*length - window) accounts for sequence length and window overlap
    return bias / (2 * len(cds1) - window)


class CDS:
    """
    A class for handling and optimizing coding sequences (CDS) with focus on codon usage
    and sequence characteristics.

    Key Features:
    - Codon optimization using CAI (Codon Adaptation Index)
    - Codon pair bias (CPB) optimization using simulated annealing
    - Sequence characteristic analysis (TA/CG content)
    - Support for species-specific optimizations
    """

    def __init__(self, cds, spe='Human', jid=0):
        """
        Initialize a CDS object with input validation and preprocessing.
        """
        # Input validation and preprocessing
        cds = cds.strip().upper()  # Normalize input sequence

        # Validate sequence length (must be codons - multiples of 3)
        if len(cds) % 3 != 0:
            raise ValueError('The length of the cds is not divisible by 3')

        # Validate sequence composition (only valid nucleotides)
        if not bool(re.match("^[atgcuATGCU]+$", cds)):
            raise ValueError('The CDS contains invalid characters')

        # Validate species exists in configuration
        if cps_dict.get(spe) is None:
            raise ValueError('The species is not in the cps_dict')

        # Load species-specific parameters
        self.cps = cps_dict[spe]        # Codon pair scores
        self.cai_logw = cai_lnw[spe]    # CAI log weights
        self.cai_choice = cai_choice[spe]  # Preferred codons

        # Process sequence
        self.cds = self.get_mrna(cds)   # Convert to standardized format
        self.jid = jid                   # Job ID for tracking

        # Calculate initial sequence characteristics
        self.init_cpb = self.cpb_calculate()  # Initial codon pair bias
        self.init_t3a1 = self.calculate_t3a1()  # Initial TA count
        self.init_c3g1 = self.calculate_c3g1()  # Initial CG count

        # Normalize metrics by sequence length
        codon_count = len(self.cds) // 3 - 1
        self.cpb = self.init_cpb / codon_count
        self.t3a1 = self.init_t3a1 / codon_count
        self.c3g1 = self.init_c3g1 / codon_count

        # Calculate CAI
        self.cai = self.calculate_cai()

        # Initialize optimization results storage
        self.cpb_query = None  # Store CPB optimization progress
        self.cpb_res = None    # Store CPB optimization results
        self.cai_res = None    # Store CAI optimization results
        self.changable_pos = None  # Track modifiable positions


def get_cai_res(self, cds=None, partial_score=1):
    """
    Optimizes a sequence by selecting preferred codons based on Codon Adaptation Index.

    This method iterates through the sequence and potentially replaces each codon
    with a preferred synonymous codon based on species-specific preferences.

    Args:
        cds (str, optional): Input coding sequence. Uses self.cds if None
        partial_score (float): Probability (0-1) of modifying each codon.
                             1 = modify all codons, 0 = modify none

    Returns:
        self: For method chaining

    Example:
        >>> cds_obj = CDS("ATGGCGTAA")
        >>> optimized = cds_obj.get_cai_res(partial_score=0.8)
    """
    # Use default sequence if none provided
    if cds is None:
        cds = self.cds

    cds = list(cds)  # Convert to list for efficient modification

    # Process each codon
    for i in range(len(cds) // 3 - 1):
        # Randomly decide whether to modify this codon
        if random.random() <= partial_score:
            # Get current amino acid and replace with preferred codon
            current_codon = ''.join(cds[i * 3:i * 3 + 3])
            amino_acid = codon_table[current_codon]
            cds[i * 3:i * 3 + 3] = self.cai_choice[amino_acid]

    self.cai_res = ''.join(cds)
    return self


def calculate_cai(self, cds=None):
    """
    Calculates the Codon Adaptation Index (CAI) for a sequence.

    CAI measures how well codon usage matches the preferred usage pattern
    of highly expressed genes in the species.

    Args:
        cds (str, optional): Sequence to analyze. Uses self.cds if None

    Returns:
        float: CAI score (geometric mean of relative adaptiveness values)

    Note:
        Uses pre-computed log weights (self.cai_logw) for efficiency
    """
    if cds is None:
        cds = self.cds

    # Sum log weights for each codon
    res = 0
    for i in range(0, len(cds) - 2, 3):
        res += self.cai_logw[cds[i:i + 3]]

    # Calculate geometric mean by averaging logs and taking exp
    res = res / (len(cds) // 3)
    return math.exp(res)


def get_cpb_res(self, res_num=100, partial_score=None, partial_mode='left',
                simulated_annealing_t0=3000, simulated_annealing_t_final=0.001,
                simulated_annealing_alpha=0.95, simulated_annealing_inner_iter=200):
    """
    Optimizes sequence for Codon Pair Bias using simulated annealing.

    Args:
        res_num (int): Number of top results to return
        partial_score (float, optional): Fraction of sequence to optimize
        partial_mode (str): How to select optimizable positions:
            - 'left': Use first X% of sequence
            - 'random': Randomly select X% of positions
        simulated_annealing_t0 (float): Initial temperature
        simulated_annealing_t_final (float): Final temperature
        simulated_annealing_alpha (float): Cooling rate (0-1)
        simulated_annealing_inner_iter (int): Iterations per temperature

    Returns:
        self: For method chaining

    Raises:
        ValueError: If partial_mode is invalid
    """
    # Reset changeable positions
    self.changable_pos = None

    # Set up partial optimization if requested
    if partial_score is not None:
        if partial_mode == 'left':
            # Use first X% of codons
            self.changable_pos = list(range(int(len(self.cds) // 3 * partial_score)))
        elif partial_mode == 'random':
            # Randomly select X% of codons
            self.changable_pos = random.sample(
                range(len(self.cds) // 3),
                int(len(self.cds) // 3 * partial_score)
            )
        else:
            raise ValueError('Invalid partial_mode')

    # Set simulated annealing parameters
    self.simulated_annealing_t0 = simulated_annealing_t0
    self.simulated_annealing_t_final = simulated_annealing_t_final
    self.simulated_annealing_alpha = simulated_annealing_alpha
    self.simulated_annealing_inner_iter = simulated_annealing_inner_iter
    self.res_num = res_num

    # Initialize optimization structures
    self.swap_dict = {}
    self.get_swap_dict()
    self.init_str = self.cds
    self._q = []          # Priority queue for top results
    self._search = []     # Track all accepted changes
    self._query = []      # Track progress

    # Run optimization
    self.cpb_res = self.simulated_annealing()
    self.cpb_query = self._query
    return self


def random_swap(self, cds=None, patical_score=1):
    """
    Performs random codon swaps throughout the sequence.

    Args:
        cds (str, optional): Sequence to modify. Uses self.cds if None
        patical_score (float): Probability of attempting swap at each position

    Returns:
        str: Modified sequence
    """
    cds = self.cds if cds is None else cds

    # Initialize swap dictionary
    self.swap_dict = {}
    self.get_swap_dict()
    self.init_str = self.cds if cds is None else cds

    # Track modified positions
    chaged = []

    # Try to swap each codon
    for aa in range(len(cds) // 3):
        # Skip if position can't be swapped
        if aa not in self.swap_dict:
            continue
        # Skip if already changed
        if aa in chaged:
            continue
        # Randomly decide whether to attempt swap
        if random.random() > patical_score:
            continue

        # Perform swap with random partner
        bb = random.choice(self.swap_dict[aa])
        chaged.append(aa)
        chaged.append(bb)
        self.swap_cabon(aa, bb)

    return self.init_str


def get_mrna(self, cds):
    """
    Standardizes RNA/DNA sequence format.

    Args:
        cds (str): Input coding sequence

    Returns:
        str: Standardized sequence (DNA format, uppercase)
    """
    return cds.replace('U', 'T').upper()


def get_swap_dict(self):
    """
    Creates a dictionary of valid codon positions that can be swapped.

    This method:
    1. Groups codons by amino acid
    2. Identifies valid swap partners (different codons for same amino acid)
    3. Respects changeable positions constraints if specified

    Returns:
        self: For method chaining

    Note:
        Creates two key data structures:
        - ammo: Groups codon positions by amino acid
        - swap_dict: Maps each position to valid swap partners
    """
    self.swap_dict = {}
    ammo = {}  # Dictionary to group positions by amino acid

    # First pass: Group positions by amino acid
    for i in range(len(self.cds) // 3):
        acid = codon_table[self.cds[i * 3:i * 3 + 3]]
        if acid not in ammo:
            ammo[acid] = []
        ammo[acid].append(i)

    # Second pass: Build swap dictionary
    for i in range(len(self.cds) // 3):
        acid = codon_table[self.cds[i * 3:i * 3 + 3]]
        for j in ammo[acid]:
            # Check if codons are different and positions are changeable
            if (self.cds[i * 3:i * 3 + 3] != self.cds[j * 3:j * 3 + 3]) and \
               (self.changable_pos is None or (i in self.changable_pos and j in self.changable_pos)):
                if i not in self.swap_dict:
                    self.swap_dict[i] = []
                self.swap_dict[i].append(j)

    return self


def get_swap_str(self, aa, bb):
    """
    Creates a new sequence with two codons swapped.

    Args:
        aa (int): First codon position (will be multiplied by 3 for nucleotide position)
        bb (int): Second codon position (will be multiplied by 3 for nucleotide position)

    Returns:
        str: New sequence with swapped codons
    """
    aa = aa * 3  # Convert to nucleotide position
    bb = bb * 3
    tmp1 = list(self.init_str)
    tmp1[aa:aa + 3] = list(self.init_str[bb:bb + 3])  # Swap first codon
    tmp1[bb:bb + 3] = list(self.init_str[aa:aa + 3])  # Swap second codon
    return ''.join(tmp1)


def swap_cabon(self, aa, bb):
    """
    Performs codon swap and updates the swap dictionary to maintain consistency.

    Args:
        aa (int): First codon position
        bb (int): Second codon position

    Side effects:
        - Updates self.init_str with new sequence
        - Updates swap_dict to reflect new positions
    """
    # Perform the actual swap
    self.init_str = self.get_swap_str(aa, bb)

    # Update swap partners for all affected positions
    for ii in self.swap_dict[aa]:
        if ii != bb:
            self.swap_dict[ii].remove(aa)
            self.swap_dict[ii].append(bb)

    for ii in self.swap_dict[bb]:
        if ii != aa:
            self.swap_dict[ii].remove(bb)
            self.swap_dict[ii].append(aa)

    # Update the swapped positions
    self.swap_dict[aa].remove(bb)
    self.swap_dict[aa].append(aa)
    self.swap_dict[bb].remove(aa)
    self.swap_dict[bb].append(bb)

    # Swap the entire lists for the two positions
    self.swap_dict[aa], self.swap_dict[bb] = self.swap_dict[bb], self.swap_dict[aa]


def calculate_cbp_change(self, aa, bb):
    """
    Calculates the change in Codon Pair Bias (CPB) score for a potential swap.

    This method efficiently calculates the CPB change by only considering
    affected regions rather than recalculating the entire sequence score.

    Args:
        aa, bb (int): Codon positions to be swapped (will be multiplied by 3)

    Returns:
        float: The change in CPB score (new_score - old_score)

    Note:
        Handles special cases:
        - Adjacent codons
        - Start of sequence (aa = 0)
        - End of sequence
    """
    # Ensure ordered positions for consistent calculation
    if aa > bb:
        return self.calculate_cbp_change(bb, aa)

    aa = 3 * aa  # Convert to nucleotide positions
    bb = 3 * bb
    old = 0  # Original score
    new = 0  # Score after swap

    # Handle adjacent codons specially
    if bb - aa == 3:
        return self.cpb_calculate(self.get_swap_str(int(aa / 3), int(bb / 3))) - \
            self.cpb_calculate(self.init_str)

    # Handle other cases
    if aa == 0:  # Start of sequence
        old += self.cps[self.init_str[aa:aa + 6]]
        new += self.cps[self.init_str[bb:bb + 3] + self.init_str[aa + 3:aa + 6]]

        if bb == len(self.init_str) - 3:  # End of sequence
            new += self.cps[self.init_str[bb - 3:bb] + self.init_str[aa:aa + 3]]
            old += self.cps[self.init_str[bb - 3:bb + 3]]
        else:
            new += self.cps[self.init_str[bb - 3:bb] + self.init_str[aa:aa + 3]] + \
                self.cps[self.init_str[aa:aa + 3] + self.init_str[bb + 3:bb + 6]]
            old += self.cps[self.init_str[bb - 3:bb + 3]] + \
                self.cps[self.init_str[bb:bb + 6]]
    else:  # Middle of sequence
        old += self.cps[self.init_str[aa - 3:aa + 3]] + \
            self.cps[self.init_str[aa:aa + 6]]
        new += self.cps[self.init_str[aa - 3:aa] + self.init_str[bb:bb + 3]] + \
            self.cps[self.init_str[bb:bb + 3] + self.init_str[aa + 3:aa + 6]]

        if bb == len(self.init_str) - 3:  # End of sequence
            old += self.cps[self.init_str[bb - 3:bb + 3]]
            new += self.cps[self.init_str[bb - 3:bb] + self.init_str[aa:aa + 3]]
        else:
            old += self.cps[self.init_str[bb - 3:bb + 3]] + \
                self.cps[self.init_str[bb:bb + 6]]
            new += self.cps[self.init_str[bb - 3:bb] + self.init_str[aa:aa + 3]] + \
                self.cps[self.init_str[aa:aa + 3] + self.init_str[bb + 3:bb + 6]]

    return new - old


def get_random_change(self):
    """
    Selects random codon positions for potential swapping in the optimization process.

    This method:
    1. Ensures enough changeable positions exist
    2. Randomly selects a codon position
    3. Randomly selects another codon position that can be swapped with the first

    Returns:
        tuple: (aa, bb) where aa and bb are codon positions that can be swapped

    Raises:
        ValueError: If there aren't enough changeable positions (<=3) in the sequence

    Note: Previous implementation using while loop is commented out in favor of
          direct selection from pre-computed swap dictionary
    """
    # Validate minimum number of changeable positions
    if len(self.swap_dict.keys()) <= 3:
        raise ValueError('No enough changeable amino acids position found.')

    # Select random position from available positions
    aa = random.choice(list(self.swap_dict.keys()))
    # Select random swap partner for selected position
    bb = random.choice(self.swap_dict[aa])

    return aa, bb


def cpb_calculate(self, cds=None):
    """
    Calculates the Codon Pair Bias (CPB) score for a given sequence.

    CPB is calculated by summing the scores of consecutive codon pairs
    using species-specific scoring matrix (self.cps).

    Args:
        cds (str, optional): The coding sequence to analyze.
                            Uses self.cds if None.

    Returns:
        float: The total CPB score for the sequence

    Note:
        - Processes sequence in 6-nucleotide windows (2 codons)
        - Uses pre-computed codon pair scores from self.cps
    """
    if cds is None:
        cds = self.cds

    res = 0
    # Sum CPB scores for each consecutive codon pair
    for ii in range(0, len(cds) - 5, 3):  # -5 to ensure full codon pairs
        res += self.cps[cds[ii:ii + 6]]  # Score for current codon pair

    return res


def metropolis(self, e, new_e, t):
    """
    Implements the Metropolis criterion for simulated annealing.

    This criterion determines whether to accept a new state in the
    simulated annealing process based on:
    1. Always accept improvements
    2. Accept some worse solutions based on:
       - How much worse they are
       - Current temperature

    Args:
        e (float): Current energy (score)
        new_e (float): New proposed energy (score)
        t (float): Current temperature

    Returns:
        bool: True if the change should be accepted, False otherwise

    Note:
        - Higher temperatures increase acceptance of worse solutions
        - As temperature decreases, becomes more selective
        - Probability of accepting worse solution: p = exp((e - new_e) / t)
    """
    # Always accept improvements
    if new_e < e:
        return True
    else:
        # Calculate acceptance probability for worse solutions
        p = math.exp((e - new_e) / t)
        # Accept based on probability
        return True if random.random() < p else False


def simulated_annealing(self):
    """
    Implements simulated annealing algorithm for codon pair bias optimization.

    The algorithm iteratively optimizes the coding sequence by:
    1. Making random codon swaps
    2. Accepting improvements
    3. Sometimes accepting suboptimal changes based on temperature

    Parameters are set through class attributes:
        - simulated_annealing_t0: Initial temperature
        - simulated_annealing_t_final: Final temperature
        - simulated_annealing_alpha: Cooling rate
        - simulated_annealing_inner_iter: Iterations at each temperature
        - res_num: Number of top results to keep

    Returns:
        list: List of dictionaries containing optimized sequences and their scores
              [{'score': float, 'cds': str}, ...]
    """
    # Initialize with original sequence
    self.init_str = self.cds

    # Get annealing parameters
    t0 = self.simulated_annealing_t0        # Starting temperature
    t_final = self.simulated_annealing_t_final  # Ending temperature
    alpha = self.simulated_annealing_alpha   # Cooling rate (0-1)
    inner_iter = self.simulated_annealing_inner_iter  # Iterations per temperature
    num = self.res_num  # Number of results to keep

    t = t0  # Current temperature
    exists = {}  # Track unique sequences
    cpb = self.init_cpb  # Initial codon pair bias score

    # Temperature reduction loop
    while t > t_final:
        # Multiple attempts at current temperature
        for i in range(inner_iter):
            # Get random codon positions to swap
            aa, bb = self.get_random_change()

            # Calculate new CPB score after potential swap
            new_cbp = cpb + self.calculate_cbp_change(aa, bb)

            # Decide whether to accept the change
            if self.metropolis(cpb, new_cbp, t):
                # Perform the swap
                self.swap_cabon(aa, bb)
                cpb = new_cbp

                # If this is a new sequence
                if exists.get(self.init_str) is None:
                    exists[self.init_str] = 1

                    # Maintain top N results using heap queue
                    if len(self._q) >= num:
                        heapq.heappop(self._q)  # Remove worst result
                    heapq.heappush(self._q, (-cpb, self.init_str))  # Add new result

                # Track all accepted changes
                self._search.append(cpb)

        # Cool down temperature
        t = alpha * t

        # Track normalized CPB score for this temperature
        self._query.append(cpb / (len(self.cds) // 3 - 1))

    # Format results for return
    res = []
    for r in self._q:
        res.append({
            'score': -r[0],  # Convert back from negative (heap was max-heap)
            'cds': r[1]
        })

    # Plotting code (commented out)
    # plt.plot(self._query)
    # plt.ylabel('CBP')
    # plt.xlabel('Iteration')
    # plt.title('CBP of simulated annealing')
    # plt.savefig('cpb_' + str(self.jid) + '.png')
    # plt.close()

    return res

    def calculate_c3g1(self):
        """Count CG dinucleotides at codon position 3-1"""
        res = 0
        for i in range(2, len(self.cds) - 2, 3):
            if self.cds[i:i + 2] == 'CG':
                res += 1
        return res

    def calculate_t3a1(self):
        """Count TA dinucleotides at codon position 3-1"""
        res = 0
        for i in range(2, len(self.cds) - 2, 3):
            if self.cds[i:i + 2] == 'TA':
                res += 1
        return res
