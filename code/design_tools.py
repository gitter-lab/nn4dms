import numpy as np
import random
import matplotlib.pyplot as plt

def generate_all_point_mutants(WT,AA_options):
    # generate every possible single point mutant
    all_mutants = []
    for pos in range(len(WT)):
        for aa in AA_options[pos]:
            if WT[pos] != aa: # not mutating to self
                mut = WT[pos]+str(pos)+aa
                all_mutants.append(mut)
    return all_mutants


def mut2seq(WT, mutations):
    mutant_seq = WT
    if type(mutations) is str:
        mutations = mutations.split(',')
    for mut in mutations:
        pos = int(mut[1:-1])
        newAA = mut[-1]
        if mut[0] != WT[pos]: print('Warning: WT residue in mutation %s does not match WT sequence' % mut)
        mutant_seq = mutant_seq[:pos] + newAA + mutant_seq[pos+1:]
    return mutant_seq


def find_top_n_mutations(seq2fitness,all_mutants,WT,n=10):
    
    # evaluate fitness of all single mutants from WT
    single_mut_fitness = []
    for mut in all_mutants:
        seq = mut2seq(WT,(mut,))
        fit = seq2fitness(seq)
        single_mut_fitness.append((mut,fit))

    # find the best mutation per position
    best_mut_per_position = []
    for pos in range(len(WT)):
        best_mut_per_position.append(max([m for m in single_mut_fitness if int(m[0][1:-1])==pos],key = lambda x: x[1]))
    
    # take the top n
    sorted_by_fitness = sorted(best_mut_per_position, key = lambda x: x[1], reverse=True)
    topn = [m[0] for m in sorted_by_fitness[:n]]
    topn = tuple([n[1] for n in sorted([(int(m[1:-1]),m) for m in topn])]) # sort by position
    
    return topn


def generate_random_mut(WT, AA_options, num_mut):
    # Want to make the probability of getting any mutation the same:
    AA_mut_options = []
    for WT_AA, AA_options_pos in zip(WT, AA_options):
        if WT_AA in AA_options_pos:
            options = list(AA_options_pos).copy()
            options.remove(WT_AA)
            AA_mut_options.append(options)
    mutations = []

    for n in range(num_mut):
        
        num_mut_pos = sum([len(row) for row in AA_mut_options])
        prob_each_pos = [len(row)/num_mut_pos for row in AA_mut_options]
        rand_num = random.random()
        for i, prob_pos in enumerate(prob_each_pos):
            rand_num -= prob_pos
            if rand_num <= 0:
                mutations.append(WT[i]+str(i)+random.choice(AA_mut_options[i]))
                AA_mut_options.pop(i)
                AA_mut_options.insert(i, [])
                break
    return ','.join(mutations)


class Hill_climber():

    def __init__(self, seq2fitness, WT, AA_options, num_mut, mut_rate=1, num_restarts=100, max_steps=1000, 
                 seq2fitness_many=None, start_seed=0):
        self.seq2fitness = seq2fitness
        self.WT = WT
        self.AA_options = AA_options
        self.num_mut = num_mut
        self.mut_rate = mut_rate
        self.num_restarts = num_restarts
        self.max_steps = max_steps
        self.seq2fitness_many = seq2fitness_many
        self.start_seed = start_seed
    
    def optimize(self):
        if self.seq2fitness_many is None or self.seq2fitness_many([self.WT]*5) is None:
            def seq2fitness_many(seqs):
                return [self.seq2fitness(seq) for seq in seqs]
            self.seq2fitness_many = seq2fitness_many
    
        fit = self.seq2fitness(self.WT)
        best_best_muts = [self.WT[0]+'0'+self.WT[0], fit]
        self.fitness_trajectory = []

        for restart in range(self.num_restarts): # At each restart, select a random set of num_mut mutations to begin from
            print('Beginning restart '+str(restart))
            random.seed(self.start_seed+restart)
            start_mut = generate_random_mut(self.WT, self.AA_options, self.num_mut)
            best_muts = [start_mut, self.seq2fitness(mut2seq(self.WT, start_mut))]
            best_fitnesses = []
            
            for step in range(self.max_steps): # Set max number of steps in hill-climbing
                neighbors = []
                current_muts = best_muts[0]
                best_fitnesses.append(self.seq2fitness(mut2seq(self.WT, current_muts)))
                # Find all neighbors (where one mutations has been deleted and one added)
                for mut in current_muts.split(','):
                    for mut_pos in range(len(self.WT)):
                        if mut_pos not in [mut[1:-1] for mut in current_muts.split(',')]:
                            AA_options_pos = [AA for AA in self.AA_options[mut_pos] if AA is not self.WT[mut_pos]]
                            new_neighbor_muts = [self.WT[mut_pos]+str(mut_pos)+AA for AA in AA_options_pos]
                            new_neighbors = [current_muts.replace(mut, new_mut) for new_mut in new_neighbor_muts]
                            neighbors.extend(new_neighbors)
                # Predict the fitnesses for all neighbors
                pred_functions = list(self.seq2fitness_many([mut2seq(self.WT, muts) for muts in neighbors]))
                best_ind = pred_functions.index(max(pred_functions))
                print('Current fitness: '+str(round(pred_functions[best_ind], 4)))
                if pred_functions[best_ind] == best_muts[1] or neighbors[best_ind] == best_muts[0]:
                    break
                best_muts = [neighbors[best_ind], pred_functions[best_ind]]
            self.fitness_trajectory.append(best_fitnesses)
            if best_muts[1] > best_best_muts[1]:
                best_best_muts = best_muts.copy()
        self.best_seq = [best_best_muts[0], best_best_muts[1]]
        return self.best_seq
        
    def plot_trajectory(self, savefig_name=None):
        for traj in self.fitness_trajectory:
            plt.plot(traj)
        plt.xlabel('Step')
        plt.ylabel('Fitness')
        plt.legend(['Restart '+str(i) for i in range(self.num_restarts)])
        if savefig_name is None:
            plt.show()
            plt.close()
        else:
            plt.savefig(savefig_name)
