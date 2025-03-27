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

with open('../cocovax/cps_dict.json', 'r') as f:
    cps_dict = json.load(f)
with open('../cocovax/cai_lnw.json', 'r') as f:
    cai_lnw = json.load(f)
with open('../cocovax/cai_choice.json', 'r') as f:
    cai_choice = json.load(f)
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
dict_ammo = {'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
             'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'], 'I': ['ATT', 'ATC', 'ATA'],
             'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], 'G': ['GGT', 'GGC', 'GGA', 'GGG'],
             'V': ['GTT', 'GTC', 'GTA', 'GTG'], 'T': ['ACT', 'ACC', 'ACA', 'ACG'], 'P': ['CCT', 'CCC', 'CCA', 'CCG'],
             'N': ['AAT', 'AAC'], 'D': ['GAT', 'GAC'], 'C': ['TGT', 'TGC'], 'Q': ['CAA', 'CAG'], 'E': ['GAA', 'GAG'],
             'H': ['CAT', 'CAC'], 'K': ['AAA', 'AAG'], 'F': ['TTT', 'TTC'], 'Y': ['TAT', 'TAC'], 'M': ['ATG'],
             'W': ['TGG'], 'STOP': ['TAG', 'TGA', 'TAA']}
transcription_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

# print(cps_dict.keys())


def get_windows(cds):
    return max(len(cds) // 5, 100)


def rna_fold(cds):
    fc = RNA.fold_compound(cds)
    (ss, mfe) = fc.mfe()
    return ss, mfe


def get_free_energy(cds):
    return rna_fold(cds)[1]


def get_mutation_distance(s1, s2):
    return edlib.align(s1, s2, mode="NW")['editDistance']


def free_energy_frame(cds, window=None):
    if window is None:
        window = get_windows(cds)
    free_energy = []
    x_index = []
    for i in range(window, len(cds) - window //
                   2 + 1, int(window // 2)):
        free_energy.append(get_free_energy(cds[i - window:min(i, len(cds))]))
        x_index.append(str(int(i - window // 2)))
    return free_energy, x_index
    # plt.figure(figsize=(10, 7))
    # plt.bar(x=x_index, height=free_energy, width=1)
    # plt.xticks(rotation=60)
    # plt.xlabel('Position')
    # plt.ylabel('Free Energy')
    # plt.title('Free Energy of mRNA', fontsize=15)
    # x_num = np.arange(len(x_index))
    # plt.xlim(min(x_num) - 0.5, max(x_num) + 0.5)
    # plt.ylim(min(free_energy) - 20, 0)
    # plt.savefig('fgf_' + str(self.jid) + '.png')
    # plt.close()


def free_energy_bias(cds1, cds2, window=None):
    if window is None:
        window = get_windows(cds1)
    if len(cds1) != len(cds2):
        raise ValueError('The length of the two CDSs are not equal')
    free_energy1 = free_energy_frame(cds1, window)[0]
    free_energy2 = free_energy_frame(cds2, window)[0]
    bias = 0
    for i in range(len(free_energy1)):
        bias += abs(free_energy1[i] - free_energy2[i])
    return bias / (2 * len(cds1) - window)


def convert_codon_ranges_to_amino_acid_ranges(codon_range_string, nt_length):
    if not codon_range_string or codon_range_string.strip() == '':
        return []
    amino_acid_ranges = []
    ranges = codon_range_string.split(',')
    for range_str in ranges:
        start, end = map(int, range_str.split(':'))
        if start < 0:
            start = start + nt_length + 1
        if end < 0:
            end = end + nt_length + 1
        aa_start = (start - 1) // 3
        aa_end = (end - 1) // 3 + 1
        amino_acid_ranges.append((aa_start, aa_end))
    if not amino_acid_ranges:
        return []
    return amino_acid_ranges


class CDS:
    def __init__(self, cds, target=None, spe='Human', jid=0):
        """
        Initializes an instance of the class.

        Arguments:
            cds (str): The CDS sequence.

        Keyword Arguments:
            spe (str): The species name. (default: 'Human')
            jid (int): The job ID. (default: 0)

        Raises:
            ValueError: If the length of the cds is not divisible by 3.
            ValueError: If the CDS contains invalid characters.
            ValueError: If the species is not in the cps_dict.
        """
        cds = cds.strip().upper()
        if len(cds) % 3 != 0:
            raise ValueError('The length of the cds is not divisible by 3')
        if not bool(re.match("^[atgcuATGCU]+$", cds)):
            raise ValueError('The CDS contains invalid characters')
        if cps_dict.get(spe) is None:
            raise ValueError('The species is not in the cps_dict')
        self.cps = cps_dict[spe]
        self.cai_logw = cai_lnw[spe]
        self.cai_choice = cai_choice[spe]
        self.cds = self.get_mrna(cds)
        self.jid = jid
        self.init_cpb = self.cpb_calculate()
        self.init_t3a1 = self.calculate_t3a1()
        self.init_c3g1 = self.calculate_c3g1()
        self.cpb = self.init_cpb / (len(self.cds) // 3 - 1)
        self.t3a1 = self.init_t3a1 / (len(self.cds) // 3 - 1)
        self.c3g1 = self.init_c3g1 / (len(self.cds) // 3 - 1)
        self.cai = self.calculate_cai()
        self.cpb_query = None
        self.cpb_res = None
        self.cai_res = None
        self.changable_pos = None
        self.target = target

    def get_cai_res(self, cds=None, partial_score=1):
        if cds is None:
            cds = self.cds
        cds = list(cds)
        for i in range(len(cds) // 3 - 1):
            if random.random() <= partial_score:
                cds[i * 3:i * 3 +
                    3] = self.cai_choice[codon_table[''.join(cds[i * 3:i * 3 + 3])]]
        self.cai_res = ''.join(cds)
        return self

    def calculate_cai(self, cds=None):
        if cds is None:
            cds = self.cds
        res = 0
        for i in range(0, len(cds) - 2, 3):
            res += self.cai_logw[cds[i:i + 3]]
        res = res / (len(cds) // 3)
        return math.exp(res)

    def get_cpb_res(self, res_num=100,
                    partial_score=100, partial_mode='left', deoptimizaion=True, simulated_annealing_t0=3000,
                    simulated_annealing_t_final=0.001,
                    simulated_annealing_alpha=0.95,
                    simulated_annealing_inner_iter=150,
                    ):
        self.changable_pos = None
        if partial_score is not None:
            if partial_mode == 'left':
                self.changable_pos = list(
                    range(int(len(self.cds) // 3 * partial_score)))
            elif partial_mode == 'random':
                self.changable_pos = random.sample(
                    range(len(self.cds) // 3), int(len(self.cds) // 3 * partial_score))
            else:
                aa_range = convert_codon_ranges_to_amino_acid_ranges(partial_mode, len(self.cds))
                self.changable_pos = []
                for aa in aa_range:
                    self.changable_pos.extend(list(range(aa[0], aa[1])))
        self.simulated_annealing_t0 = simulated_annealing_t0
        self.simulated_annealing_t_final = simulated_annealing_t_final
        self.simulated_annealing_alpha = simulated_annealing_alpha
        self.simulated_annealing_inner_iter = simulated_annealing_inner_iter
        self.res_num = res_num
        self.swap_dict = {}
        self.get_swap_dict()
        self.init_str = self.cds
        self._q = []
        self._query_seq = []
        self._query = []
        self._query_mut = []
        self.init_str = self.cds
        self.cpb_res = self.simulated_annealing()
        self.cpb_query = self._query
        return self

    def random_swap(self, cds=None, patical_score=1):
        cds = self.cds if cds is None else cds
        self.swap_dict = {}
        self.get_swap_dict()
        self.init_str = self.cds if cds is None else cds
        chaged = []
        for aa in range(len(cds) // 3):
            if aa not in self.swap_dict:
                continue
            if aa in chaged:
                continue
            if random.random() > patical_score:
                continue
            bb = random.choice(self.swap_dict[aa])
            chaged.append(aa)
            chaged.append(bb)
            self.swap_cabon(aa, bb)
        return self.init_str

    def get_mrna(self, cds):
        # return ''.join(list(map(lambda x: transcription_dict[x],
        # string.upper())))
        return cds.replace('U', 'T').upper()

    def get_swap_dict(self):
        self.swap_dict = {}
        ammo = {}
        for i in range(len(self.cds) // 3):
            acid = codon_table[self.cds[i * 3:i * 3 + 3]]
            if acid not in ammo:
                ammo[acid] = []
            ammo[acid].append(i)
        for i in range(len(self.cds) // 3):
            acid = codon_table[self.cds[i * 3:i * 3 + 3]]
            for j in ammo[acid]:
                if (self.cds[i * 3:i * 3 + 3] != self.cds[j * 3:j * 3 + 3]) and (self.changable_pos is None or (
                        i in self.changable_pos and j in self.changable_pos)):
                    if i not in self.swap_dict:
                        self.swap_dict[i] = []
                    self.swap_dict[i].append(j)
                    # self.swap_dict[i].append(j)
        return self

    def get_swap_str(self, aa, bb):
        aa = aa * 3
        bb = bb * 3
        tmp1 = list(self.init_str)
        tmp1[aa:aa + 3] = list(self.init_str[bb:bb + 3])
        tmp1[bb:bb + 3] = list(self.init_str[aa:aa + 3])
        return ''.join(tmp1)

    def swap_cabon(self, aa, bb):
        # print(swap_dict[aa])
        # print(swap_dict[bb])
        self.init_str = self.get_swap_str(aa, bb)
        for ii in self.swap_dict[aa]:
            if ii != bb:
                self.swap_dict[ii].remove(aa)
                self.swap_dict[ii].append(bb)
        for ii in self.swap_dict[bb]:
            if ii != aa:
                self.swap_dict[ii].remove(bb)
                self.swap_dict[ii].append(aa)
        self.swap_dict[aa].remove(bb)
        self.swap_dict[aa].append(aa)
        self.swap_dict[bb].remove(aa)
        self.swap_dict[bb].append(bb)
        self.swap_dict[aa], self.swap_dict[bb] = self.swap_dict[bb], self.swap_dict[aa]
        # print(swap_dict[aa])
        # print(swap_dict[bb])

    def calculate_cbp_change(self, aa, bb):
        """
        :param aa: amino acid position aa
        :param bb: amino acid position bb
        :return: change of cbp
        """
        if aa > bb:
            return self.calculate_cbp_change(bb, aa)
        aa = 3 * aa
        bb = 3 * bb
        old = 0
        new = 0
        if bb - aa == 3:
            return self.cpb_calculate(self.get_swap_str(int(
                aa / 3), int(bb / 3))) - self.cpb_calculate(self.init_str)
        else:
            if aa == 0:
                old += self.cps[self.init_str[aa:aa + 6]]
                new += self.cps[self.init_str[bb:bb + 3] +
                                self.init_str[aa + 3:aa + 6]]
                if bb == len(self.init_str) - 3:
                    new += self.cps[self.init_str[bb - 3:bb] +
                                    self.init_str[aa:aa + 3]]
                    old += self.cps[self.init_str[bb - 3:bb + 3]]
                else:
                    new += self.cps[self.init_str[bb - 3:bb] + self.init_str[aa:aa + 3]] + self.cps[
                        self.init_str[aa:aa + 3] + self.init_str[bb + 3:bb + 6]]
                    old += self.cps[self.init_str[bb - 3:bb + 3]] + \
                        self.cps[self.init_str[bb:bb + 6]]
            else:
                old += self.cps[self.init_str[aa - 3:aa + 3]] + \
                    self.cps[self.init_str[aa:aa + 6]]
                new += self.cps[self.init_str[aa - 3:aa] + self.init_str[bb:bb + 3]] + self.cps[
                    self.init_str[bb:bb + 3] + self.init_str[aa + 3:aa + 6]]
                if bb == len(self.init_str) - 3:
                    old += self.cps[self.init_str[bb - 3:bb + 3]]
                    new += self.cps[self.init_str[bb - 3:bb] +
                                    self.init_str[aa:aa + 3]]
                else:
                    old += self.cps[self.init_str[bb - 3:bb + 3]] + \
                        self.cps[self.init_str[bb:bb + 6]]
                    new += self.cps[self.init_str[bb - 3:bb] + self.init_str[aa:aa + 3]] + self.cps[
                        self.init_str[aa:aa + 3] + self.init_str[bb + 3:bb + 6]]
            return new - old

    def get_random_change(self):
        # while True:
        #     aa = random.randint(0, len(self.init_str) // 3 - 1)
        #     index += 1
        #     if self.swap_dict.get(aa) is not None:
        #         break
        #     if index > len(self.init_str):
        #         raise ValueError(
        #             'No valid changeable amino acid position found.')
        if len(self.swap_dict.keys()) <= 3:
            raise ValueError(
                'No enough changeable amino acids position found.')
        aa = random.choice(list(self.swap_dict.keys()))
        bb = random.choice(self.swap_dict[aa])
        return aa, bb

    def cpb_calculate(self, cds=None):
        if cds is None:
            cds = self.cds
        res = 0
        for ii in range(0, len(cds) - 5, 3):
            res += self.cps[cds[ii:ii + 6]]
        return res

    def metropolis_deoptimization(self, e, new_e, t):
        if new_e < e:
            return True
        else:
            p = math.exp((e - new_e) / t)
            return True if random.random() < p else False

    def metropolis_optimization(self, e, new_e, t):
        if new_e > e:
            return True
        else:
            p = math.exp((new_e - e) / t)
            return True if random.random() < p else False

    def simulated_annealing(
            self):
        self.init_str = self.cds
        t0 = self.simulated_annealing_t0
        t_final = self.simulated_annealing_t_final
        alpha = self.simulated_annealing_alpha
        inner_iter = self.simulated_annealing_inner_iter
        num = self.res_num
        t = t0
        # 外层循环
        cpb = self.init_cpb
        while t > t_final:
            # 内层循环
            # print(t)
            for i in range(inner_iter):
                # print(i)
                aa, bb = self.get_random_change()
                # print(aa, bb)
                new_cbp = cpb + self.calculate_cbp_change(aa, bb)
                # print(new_cbp, self.calculate_cbp_change(aa, bb))
                if self.metropolis_deoptimization(cpb, new_cbp, t):
                    self.swap_cabon(aa, bb)
                    cpb = new_cbp
                    if len(self._q) >= num:
                        heapq.heappop(self._q)
                    heapq.heappush(self._q, (-cpb, self.init_str))
            t = alpha * t
            self._query.append(cpb / (len(self.cds) // 3 - 1))
            self._query_seq.append(self.init_str)
            self._query_mut.append(
                get_mutation_distance(self.init_str, self.target))
            self._query.append(cpb / (len(self.cds) // 3 - 1))
        res = []
        for r in self._q:
            res.append({'score': -r[0], 'cds': r[1]})
        # plt.plot(self._query)
        # plt.ylabel('CBP')
        # plt.xlabel('Iteration')
        # plt.title('CBP of simulated annealing')
        # plt.savefig('cpb_' + str(self.jid) + '.png')
        # plt.close()
        return res

    def calculate_c3g1(self):
        res = 0
        for i in range(2, len(self.cds) - 2, 3):
            if self.cds[i:i + 2] == 'CG':
                res += 1
        if res == 0:
            return 0.0001
        return res

    def calculate_t3a1(self):
        res = 0
        for i in range(2, len(self.cds) - 2, 3):
            if self.cds[i:i + 2] == 'TA':
                res += 1
        if res == 0:
            return 0.0001
        return res
