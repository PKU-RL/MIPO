import os
import random
import copy
import codecs
import spacy
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from textworld import EnvInfos
from torch.distributions import Categorical

import dqn_memory_priortized_replay_buffer
from model import KG_Manipulation
from generic import to_np, to_pt, _words_to_ids, _word_to_id, pad_sequences, update_graph_triplets, preproc, max_len, ez_gather_dim_1
from generic import sort_target_commands, process_facts, serialize_facts, gen_graph_commands, process_fully_obs_facts
from generic import generate_labels_for_ap, generate_labels_for_sp, LinearSchedule
from layers import NegativeLogLoss, compute_mask, masked_mean

import os
from transformers import AutoTokenizer, GPT2Model, set_seed
from transformers import DistilBertTokenizer, DistilBertModel
from scipy.special import softmax

def get_tokenizer_and_model():
    tokenizer = DistilBertTokenizer.from_pretrained('../lm/distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("../lm/distilbert-base-uncased")
    target_model = DistilBertModel.from_pretrained("../lm/distilbert-base-uncased")
    return tokenizer, model, target_model

class Agent:
    def __init__(self, config):
        self.mode = "train"
        self.config = config
        print(self.config)
        self.load_config()

        self.online_net = KG_Manipulation(config=self.config, word_vocab=self.word_vocab, node_vocab=self.node_vocab, relation_vocab=self.relation_vocab)
        self.prior_tokenizer, self.prior, self.prior_target = get_tokenizer_and_model()

        for param in self.prior.parameters():
            param.requires_grad = False
        for param in self.prior_target.parameters():
            param.requires_grad = False
        if self.use_prior:
            for param in self.prior.transformer.layer[-1].parameters():
                param.requires_grad = True

        self.alpha_start = 0.5
        self.alpha_min = 0.1
        self.alpha_step = (self.alpha_start - self.alpha_min) / 20000
        self.alpha = self.alpha_start
        
        self.prior_beta = 25
        
        self.online_net.train()
        self.device = torch.device('cpu')
        if self.use_cuda:
            self.device = torch.device('cuda')
            self.prior.cuda()
            self.prior_target.cuda()
            self.online_net.cuda()

        if self.task == "rl":
            self.target_net = KG_Manipulation(config=self.config, word_vocab=self.word_vocab, node_vocab=self.node_vocab, relation_vocab=self.relation_vocab)
            self.pretrained_graph_generation_net = KG_Manipulation(config=self.config, word_vocab=self.word_vocab, node_vocab=self.node_vocab, relation_vocab=self.relation_vocab)
            self.target_net.train()
            self.pretrained_graph_generation_net.eval()
            self.update_target_net()
            for param in self.target_net.parameters():
                param.requires_grad = False
            for param in self.pretrained_graph_generation_net.parameters():
                param.requires_grad = False
            if self.use_cuda:
                self.target_net.cuda()
                self.pretrained_graph_generation_net.cuda()
        else:
            self.target_net, self.pretrained_graph_generation_net = None, None

        # exclude some parameters from optimizer
        param_frozen_list = [] # should be changed into torch.nn.ParameterList()
        param_active_list = [] # should be changed into torch.nn.ParameterList()
        for k, v in self.online_net.named_parameters():
            keep_this = True
            for keyword in self.fix_parameters_keywords:
                if keyword in k:
                    param_frozen_list.append(v)
                    keep_this = False
                    break
            if keep_this:
                param_active_list.append(v)

        param_frozen_list = torch.nn.ParameterList(param_frozen_list)
        param_active_list = torch.nn.ParameterList(param_active_list)

        # optimizer
        if self.step_rule == 'adam':
            self.optimizer = torch.optim.Adam([{'params': param_frozen_list, 'lr': 0.0},
                                               {'params': param_active_list, 'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                              lr=self.config['general']['training']['optimizer']['learning_rate'])
        elif self.step_rule == 'radam':
            from radam import RAdam
            self.optimizer = RAdam([{'params': param_frozen_list, 'lr': 0.0},
                                    {'params': param_active_list, 'lr': self.config['general']['training']['optimizer']['learning_rate']}],
                                   lr=self.config['general']['training']['optimizer']['learning_rate'])

            
        else:
            raise NotImplementedError
        self.policy_optimizer = torch.optim.Adam(self.online_net.parameters(), lr=5e-4)
        self.prior_optimizer = torch.optim.Adam(self.prior.parameters(), lr=5e-4)

    def load_config(self):
        self.real_valued_graph = self.config['general']['model']['real_valued_graph']
        self.task = self.config['general']['task']
        # word vocab
        self.word_vocab = []
        with codecs.open("./vocabularies/word_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.word_vocab.append(line.strip())
        self.word2id = {}
        for i, w in enumerate(self.word_vocab):
            self.word2id[w] = i
        # node vocab
        self.node_vocab = []
        with codecs.open("./vocabularies/node_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.node_vocab.append(line.strip().lower())
        self.node2id = {}
        for i, w in enumerate(self.node_vocab):
            self.node2id[w] = i
        # relation vocab
        self.relation_vocab = []
        with codecs.open("./vocabularies/relation_vocab.txt", mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.relation_vocab.append(line.strip().lower())
        self.origin_relation_number = len(self.relation_vocab)
        # add reverse relations
        for i in range(self.origin_relation_number):
            self.relation_vocab.append(self.relation_vocab[i] + "_reverse")
        if not self.real_valued_graph:
            # add self relation
            self.relation_vocab += ["self"]
        self.relation2id = {}
        for i, w in enumerate(self.relation_vocab):
            self.relation2id[w] = i

        self.step_rule = self.config['general']['training']['optimizer']['step_rule']
        self.init_learning_rate = self.config['general']['training']['optimizer']['learning_rate']
        self.clip_grad_norm = self.config['general']['training']['optimizer']['clip_grad_norm']
        self.learning_rate_warmup_until = self.config['general']['training']['optimizer']['learning_rate_warmup_until']
        self.fix_parameters_keywords = list(set(self.config['general']['training']['fix_parameters_keywords']))
        self.batch_size = self.config['general']['training']['batch_size']
        self.max_episode = self.config['general']['training']['max_episode']
        self.smoothing_eps = self.config['general']['training']['smoothing_eps']
        self.patience = self.config['general']['training']['patience']

        self.run_eval = self.config['general']['evaluate']['run_eval']
        self.eval_g_belief = self.config['general']['evaluate']['g_belief']
        self.eval_batch_size = self.config['general']['evaluate']['batch_size']
        self.max_target_length = self.config['general']['evaluate']['max_target_length']

        # Set the random seed manually for reproducibility.
        self.random_seed = self.config['general']['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            if not self.config['general']['use_cuda']:
                print("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
                self.use_cuda = False
            else:
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(self.random_seed)
                self.use_cuda = True
        else:
            self.use_cuda = False

        self.experiment_tag = "{}_difficulty_{}_seed_{}".format(self.config['general']['checkpoint']['experiment_tag'], self.config['rl']['difficulty_level'], self.config['general']['random_seed'])
        self.save_frequency = self.config['general']['checkpoint']['save_frequency']
        self.report_frequency = self.config['general']['checkpoint']['report_frequency']
        self.load_pretrained = self.config['general']['checkpoint']['load_pretrained']
        self.load_from_tag = self.config['general']['checkpoint']['load_from_tag']
        self.load_graph_generation_model_from_tag = self.config['general']['checkpoint']['load_graph_generation_model_from_tag']
        self.load_parameter_keywords = list(set(self.config['general']['checkpoint']['load_parameter_keywords']))

        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])

        self.backprop_frequency = self.config['obs_gen']['backprop_frequency']

        # AP specific
        self.ap_k_way_classification = self.config['ap']['k_way_classification']

        # SP specific
        self.sp_k_way_classification = self.config['sp']['k_way_classification']

        # DGI specific
        self.sample_bias_positive = self.config['dgi']['sample_bias_positive']
        self.sample_bias_negative = self.config['dgi']['sample_bias_negative']

        # RL specific
        self.fully_observable_graph = self.config['rl']['fully_observable_graph']
        # epsilon greedy
        self.epsilon_anneal_episodes = self.config['rl']['epsilon_greedy']['epsilon_anneal_episodes']
        self.epsilon_anneal_from = self.config['rl']['epsilon_greedy']['epsilon_anneal_from']
        self.epsilon_anneal_to = self.config['rl']['epsilon_greedy']['epsilon_anneal_to']
        self.epsilon = self.epsilon_anneal_from
        self.epsilon_scheduler = LinearSchedule(schedule_timesteps=self.epsilon_anneal_episodes, initial_p=self.epsilon_anneal_from, final_p=self.epsilon_anneal_to)
        self.noisy_net = self.config['rl']['epsilon_greedy']['noisy_net']
        if self.noisy_net:
            # disable epsilon greedy
            self.epsilon_anneal_episodes = -1
            self.epsilon = 0.0
        # drqn
        self.replay_sample_history_length = self.config['rl']['replay']['replay_sample_history_length']
        self.replay_sample_update_from = self.config['rl']['replay']['replay_sample_update_from']
        # replay buffer and updates
        self.buffer_reward_threshold = self.config['rl']['replay']['buffer_reward_threshold']
        self.prioritized_replay_beta = self.config['rl']['replay']['prioritized_replay_beta']
        self.beta_scheduler = LinearSchedule(schedule_timesteps=self.max_episode, initial_p=self.prioritized_replay_beta, final_p=1.0)

        self.accumulate_reward_from_final = self.config['rl']['replay']['accumulate_reward_from_final']
        self.prioritized_replay_eps = self.config['rl']['replay']['prioritized_replay_eps']
        self.count_reward_lambda = self.config['rl']['replay']['count_reward_lambda']
        self.discount_gamma_count_reward = self.config['rl']['replay']['discount_gamma_count_reward']
        self.graph_reward_lambda = self.config['rl']['replay']['graph_reward_lambda']
        self.graph_reward_type = self.config['rl']['replay']['graph_reward_type']
        self.discount_gamma_graph_reward = self.config['rl']['replay']['discount_gamma_graph_reward']
        self.discount_gamma_game_reward = self.config['rl']['replay']['discount_gamma_game_reward']
        self.replay_batch_size = self.config['rl']['replay']['replay_batch_size']
        self.dqn_memory = dqn_memory_priortized_replay_buffer.PrioritizedReplayMemory(self.config['rl']['replay']['replay_memory_capacity'],
                                                                                      priority_fraction=self.config['rl']['replay']['replay_memory_priority_fraction'],
                                                                                      discount_gamma_game_reward=self.discount_gamma_game_reward,
                                                                                      discount_gamma_graph_reward=self.discount_gamma_graph_reward,
                                                                                      discount_gamma_count_reward=self.discount_gamma_count_reward,
                                                                                      accumulate_reward_from_final=self.accumulate_reward_from_final,
                                                                                      seed=self.config['general']['random_seed'])
        self.prior_memory = dqn_memory_priortized_replay_buffer.PrioritizedReplayMemory(self.config['general']['training']['batch_size'] * self.config['rl']['training']['max_nb_steps_per_episode'] * 10,
                                                                                      discount_gamma_game_reward=self.discount_gamma_game_reward,
                                                                                      discount_gamma_graph_reward=self.discount_gamma_graph_reward,
                                                                                      discount_gamma_count_reward=self.discount_gamma_count_reward,
                                                                                      accumulate_reward_from_final=self.accumulate_reward_from_final,
                                                                                      seed=self.config['general']['random_seed'])
        self.update_per_k_game_steps = self.config['rl']['replay']['update_per_k_game_steps']
        self.multi_step = self.config['rl']['replay']['multi_step']
        # input in rl training
        self.enable_recurrent_memory = self.config['rl']['model']['enable_recurrent_memory']
        self.enable_graph_input = self.config['rl']['model']['enable_graph_input']
        self.enable_text_input = self.config['rl']['model']['enable_text_input']
        assert self.enable_graph_input or self.enable_text_input
        # rl train and eval
        self.max_nb_steps_per_episode = self.config['rl']['training']['max_nb_steps_per_episode']
        self.learn_start_from_this_episode = self.config['rl']['training']['learn_start_from_this_episode']
        self.target_net_update_frequency = self.config['rl']['training']['target_net_update_frequency']
        self.use_negative_reward = self.config['rl']['training']['use_negative_reward']
        self.eval_max_nb_steps_per_episode = self.config['rl']['evaluate']['max_nb_steps_per_episode']
        self.use_prior = self.config['rl']['use_prior']

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.online_net.train()
        self.prior.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.online_net.eval()
        self.prior.eval()
        self.prior_target.eval()

    def update_target_net(self):
        if self.target_net is not None:
            self.target_net.load_state_dict(self.online_net.state_dict())
        if self.prior_target is not None:
            self.prior_target.load_state_dict(self.prior.state_dict())

    def reset_noise(self):
        if self.noisy_net:
            # Resets noisy weights in all linear layers (of online net only)
            self.online_net.reset_noise()

    def zero_noise(self):
        if self.noisy_net:
            self.online_net.zero_noise()
            if self.target_net is not None:
                self.target_net.zero_noise()
            if self.pretrained_graph_generation_net is not None:
                self.pretrained_graph_generation_net.zero_noise()

    def load_pretrained_graph_generation_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading pre-trained graph generation model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')
            try:
                self.pretrained_graph_generation_net.load_state_dict(pretrained_dict)
            except:
                # graph generation net
                model_dict = self.pretrained_graph_generation_net.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.pretrained_graph_generation_net.load_state_dict(model_dict)
                print("WARNING... Model dict is different with pretrained dict. I'm loading only the parameters with same labels now. Make sure you really want this...")
                print("The loaded parameters are:")
                keys = [key for key in pretrained_dict]
                print(", ".join(keys))
                print("--------------------------")
        except:
            print("Failed to load checkpoint...")

    def load_pretrained_model(self, load_from, load_partial_graph=True):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: File name of the pretrained model checkpoint.
        """
        print("loading model from %s\n" % (load_from))
        try:
            if self.use_cuda:
                pretrained_dict = torch.load(load_from)
            else:
                pretrained_dict = torch.load(load_from, map_location='cpu')

            model_dict = self.online_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if load_partial_graph and len(self.load_parameter_keywords) > 0:
                tmp_pretrained_dict = {}
                for k, v in pretrained_dict.items():
                    for keyword in self.load_parameter_keywords:
                        if keyword in k:
                            tmp_pretrained_dict[k] = v
                            break
                pretrained_dict = tmp_pretrained_dict
            model_dict.update(pretrained_dict)
            self.online_net.load_state_dict(model_dict)
            print("The loaded parameters are:")
            keys = [key for key in pretrained_dict]
            print(", ".join(keys))
            print("--------------------------")
        except:
            print("Failed to load checkpoint...")

    def save_model_to_path(self, save_to):
        torch.save(self.online_net.state_dict(), save_to)
        print("Saved checkpoint to %s..." % (save_to))

    def select_additional_infos(self):
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.admissible_commands = True
        request_infos.description = True
        request_infos.location = True
        request_infos.facts = True
        request_infos.last_action = True
        request_infos.game = True
        if self.use_negative_reward:
            request_infos.has_lost = True
            request_infos.has_won = True
        return request_infos

    def select_additional_infos_lite(self):
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.admissible_commands = True
        request_infos.description = False
        request_infos.location = False
        request_infos.facts = False
        request_infos.last_action = False
        request_infos.game = True
        if self.use_negative_reward:
            request_infos.has_lost = True
            request_infos.has_won = True
        return request_infos

    def init(self):
        pass

    def get_word_input(self, input_strings):
        word_list = [item.split() for item in input_strings]
        word_id_list = [_words_to_ids(tokens, self.word2id) for tokens in word_list]
        input_word = pad_sequences(word_id_list, maxlen=max_len(word_id_list)).astype('int32')
        input_word = to_pt(input_word, self.use_cuda)
        return input_word

    def get_graph_adjacency_matrix(self, triplets):
        adj = np.zeros((len(triplets), len(self.relation_vocab), len(self.node_vocab), len(self.node_vocab)), dtype="float32")
        for b in range(len(triplets)):
            node_exists = set()
            for t in triplets[b]:
                node1, node2, relation = t
                assert node1 in self.node_vocab, node1 + " is not in node vocab"
                assert node2 in self.node_vocab, node2 + " is not in node vocab"
                assert relation in self.relation_vocab, relation + " is not in relation vocab"
                node1_id, node2_id, relation_id = _word_to_id(node1, self.node2id), _word_to_id(node2, self.node2id), _word_to_id(relation, self.relation2id)
                adj[b][relation_id][node1_id][node2_id] = 1.0
                adj[b][relation_id + self.origin_relation_number][node2_id][node1_id] = 1.0
                node_exists.add(node1_id)
                node_exists.add(node2_id)
            # self relation
            for node_id in list(node_exists):
                adj[b, -1, node_id, node_id] = 1.0
        adj = to_pt(adj, self.use_cuda, type='float')
        return adj

    def get_graph_node_name_input(self):
        res = copy.copy(self.node_vocab)
        input_node_name = self.get_word_input(res)  # num_node x words
        return input_node_name

    def get_graph_relation_name_input(self):
        res = copy.copy(self.relation_vocab)
        res = [item.replace("_", " ") for item in res]
        input_relation_name = self.get_word_input(res)  # num_node x words
        return input_relation_name

    def get_action_candidate_list_input(self, action_candidate_list):
        # action_candidate_list (list): batch x num_candidate of strings
        batch_size = len(action_candidate_list)
        max_num_candidate = max_len(action_candidate_list)
        input_action_candidate_list = []
        for i in range(batch_size):
            word_level = self.get_word_input(action_candidate_list[i]) #seq of words ---> seq of ids, each of it padded to same lenth
            input_action_candidate_list.append(word_level)
        max_word_num = max([item.size(1) for item in input_action_candidate_list])

        input_action_candidate = np.zeros((batch_size, max_num_candidate, max_word_num)) # padding to same number of actions, and each action of same lenth
        input_action_candidate = to_pt(input_action_candidate, self.use_cuda, type="long")
        for i in range(batch_size):
            input_action_candidate[i, :input_action_candidate_list[i].size(0), :input_action_candidate_list[i].size(1)] = input_action_candidate_list[i]

        return input_action_candidate

    def choose_model(self, use_model="online"):
        if self.task != "rl":
            return self.online_net
        if use_model == "online":
            model = self.online_net
        elif use_model == "target":
            model = self.target_net
        elif use_model == "pretrained_graph_generation":
            model = self.pretrained_graph_generation_net
        else:
            raise NotImplementedError
        return model

    def encode_graph(self, graph_input, use_model):
        model = self.choose_model(use_model)
        input_node_name = self.get_graph_node_name_input()
        input_relation_name = self.get_graph_relation_name_input()
        if isinstance(graph_input, list):
            adjacency_matrix = self.get_graph_adjacency_matrix(graph_input)
        elif isinstance(graph_input, torch.Tensor):
            adjacency_matrix = graph_input
        else:
            raise NotImplementedError
        node_encoding_sequence, node_mask = model.encode_graph(input_node_name, input_relation_name, adjacency_matrix)
        return node_encoding_sequence, node_mask

    def encode_text(self, observation_strings, use_model):
        model = self.choose_model(use_model)
        input_obs = self.get_word_input(observation_strings) #seq of words --> seq of ids
        # encode
        obs_encoding_sequence, obs_mask = model.encode_text(input_obs)
        return obs_encoding_sequence, obs_mask

    ##################################
    # RL specific
    ##################################

    def finish_of_episode(self, episode_no, batch_size):
        # Update target network
        if (episode_no + batch_size) % self.target_net_update_frequency <= episode_no % self.target_net_update_frequency:
            self.update_target_net()
        # decay lambdas
        if episode_no < self.learn_start_from_this_episode:
            return
        if episode_no < self.epsilon_anneal_episodes + self.learn_start_from_this_episode:
            self.epsilon = self.epsilon_scheduler.value(episode_no - self.learn_start_from_this_episode)
            self.epsilon = max(self.epsilon, 0.0)
            
            self.alpha = self.alpha_start - (self.alpha_step * episode_no)
            self.alpha = max(self.alpha, self.alpha_min)

    def get_game_info_at_certain_step_fully_observable(self, obs, infos):
        """
        Get all needed info from game engine for training.
        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        batch_size = len(obs)
        #observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        observation_strings = []
        for b in range(batch_size):
            obs_str = obs[b]
            goal_str = infos["game"][b].metadata['recipe']
            observation_strings.append(preproc("your goal is: " + goal_str, tokenizer=self.nlp))

        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)

        # get new facts
        current_triplets = []  # batch of list of triplets
        for b in range(batch_size):
            new_f = set(process_fully_obs_facts(infos["game"][b], infos["facts"][b]))
            triplets = serialize_facts(new_f)
            current_triplets.append(triplets)

        return observation_strings, current_triplets, action_candidate_list, None, None

    def get_game_info_at_certain_step(self, obs, infos, prev_actions=None, prev_facts=None, return_gt_commands=False):
        """
        Get all needed info from game engine for training.
        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        if self.fully_observable_graph:
            return self.get_game_info_at_certain_step_fully_observable(obs, infos)

        batch_size = len(obs)
        observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)

        # get new facts
        new_facts = []
        current_triplets = []  # batch of list of triplets
        commands_from_env = []  # batch of list of commands
        for b in range(batch_size):
            if prev_facts is None:
                new_f = process_facts(None, infos["game"][b], infos["facts"][b], None, None)
                prev_f = set()
            else:
                new_f = process_facts(prev_facts[b], infos["game"][b], infos["facts"][b], infos["last_action"][b], prev_actions[b])
                prev_f = prev_facts[b]
            new_facts.append(new_f)
            triplets = serialize_facts(new_f)
            current_triplets.append(triplets)
            target_commands = gen_graph_commands(new_f - prev_f, cmd="add") + gen_graph_commands(prev_f - new_f, cmd="delete")
            commands_from_env.append(target_commands)

        target_command_strings = []
        if return_gt_commands:
            # sort target commands and add seperators.
            target_command_strings = [" <sep> ".join(sort_target_commands(tgt_cmds)) for tgt_cmds in commands_from_env]

        return observation_strings, current_triplets, action_candidate_list, target_command_strings, new_facts

    def get_game_info_at_certain_step_lite(self, obs, infos):
        """
        Get all needed info from game engine for training.
        Arguments:
            obs: Previous command's feedback for each game.
            infos: Additional information for each game.
        """
        if self.fully_observable_graph:
            return self.get_game_info_at_certain_step_fully_observable(obs, infos)

        batch_size = len(obs)
        observation_strings = [preproc(item, tokenizer=self.nlp) for item in obs]
        action_candidate_list = []
        for b in range(batch_size):
            ac = [preproc(item, tokenizer=self.nlp) for item in infos["admissible_commands"][b]]
            action_candidate_list.append(ac)

        return observation_strings, action_candidate_list

    def update_knowledge_graph_triplets(self, triplets, prediction_strings):
        new_triplets = []
        for i in range(len(triplets)):
            # per example in a batch
            predict_cmds = prediction_strings[i].split("<sep>")
            if predict_cmds[-1].endswith("<eos>"):
                predict_cmds[-1] = predict_cmds[-1][:-5].strip()
            else:
                predict_cmds = predict_cmds[:-1]
            if len(predict_cmds) == 0:
                new_triplets.append(triplets[i])
                continue
            predict_cmds = [" ".join(item.split()) for item in predict_cmds]
            predict_cmds = [item for item in predict_cmds if len(item) > 0]
            new_triplets.append(update_graph_triplets(triplets[i], predict_cmds, self.node_vocab, self.relation_vocab))
        return new_triplets

    def encode(self, observation_strings, graph_input, use_model):
        assert self.task == "rl"
        model = self.choose_model(use_model)
        # step 1 and 3, at step 3, the agent doesn't have to re-encode observation
        # because it's essentially the same as in step 1
        if self.enable_text_input:
            obs_encoding_sequence, obs_mask = self.encode_text(observation_strings, use_model=use_model)
        else:
            obs_encoding_sequence, obs_mask = None, None

        if self.enable_graph_input:
            node_encoding_sequence, node_mask = self.encode_graph(graph_input, use_model=use_model)
        else:
            node_encoding_sequence, node_mask = None, None

        if self.enable_text_input and self.enable_graph_input:
            h_og, h_go = model.get_match_representations(obs_encoding_sequence, obs_mask, node_encoding_sequence, node_mask)
            return h_og, obs_mask, h_go, node_mask
        else:
            return obs_encoding_sequence, obs_mask, node_encoding_sequence, node_mask

    def pad_actions_prior(self, action_candidate_list, pad_token):
        max_num_candidate = max_len(action_candidate_list)
        padded_action_candidate_list = []
        padded_mask = []
        for i in range(len(action_candidate_list)):
            for j in range(max_num_candidate):
                #cur_len = len(action_candidate_list[i])
                #padded_action_candidates = action_candidate_list[i] + [pad_token] * (max_num_candidate - cur_len)
                try:
                    padded_action_candidate_list.append(action_candidate_list[i][j])
                    padded_mask.append(1)
                except:
                    padded_action_candidate_list.append(pad_token)
                    padded_mask.append(0)
        return padded_action_candidate_list, np.array(padded_mask)


    def get_prior_probs(self, observation_strings, action_candidate_list, action_masks, use_target):
        #self.prior_tokenizer.pad_token = self.prior_tokenizer.eos_token
        if use_target:
            prior_policy = self.prior_target
        else:
            prior_policy = self.prior
        B, A = action_masks.shape
        obs_inputs = self.prior_tokenizer.batch_encode_plus(observation_strings, return_tensors = "pt", padding = True).to(device=self.device)
        obs_ids = obs_inputs['input_ids']
        obs_mask = obs_inputs['attention_mask']
        obs_outputs = prior_policy(input_ids=obs_ids, attention_mask=obs_mask)
        obs_embed = obs_outputs.last_hidden_state
        _mask = torch.sum(obs_mask, -1)
        _all_zero = torch.eq(_mask, 0).float()
        obs_embed = (obs_embed * obs_mask.unsqueeze(-1)).sum(dim=1) / (_mask + _all_zero).unsqueeze(-1) # BV

        padded_action_candidate_list, padded_masks = self.pad_actions_prior(action_candidate_list, self.prior_tokenizer.pad_token)

        padded_masks = torch.tensor(padded_masks).to(dtype=torch.float, device=self.device)
        padded_masks = padded_masks.reshape(B, -1)

        padded_action_inputs = self.prior_tokenizer.batch_encode_plus(padded_action_candidate_list, return_tensors = "pt", padding = True).to(device=self.device)
        padded_action_ids = padded_action_inputs['input_ids']
        padded_action_mask = padded_action_inputs['attention_mask']
        action_outputs = prior_policy(input_ids=padded_action_ids, attention_mask=padded_action_mask)
        action_embed = action_outputs.last_hidden_state # BAV
        _mask = torch.sum(padded_action_mask, -1)
        _all_zero = torch.eq(_mask, 0).float()
        action_embed = (action_embed * padded_action_mask.unsqueeze(-1)).sum(dim=1) / (_mask + _all_zero).unsqueeze(-1) #(B*A)V
        action_embed = action_embed.reshape(B,A,-1)
        action_embed = action_embed / ((((action_embed ** 2).sum(dim=-1, keepdim=True)) ** 0.5).detach())


        tiled_obs_embed = obs_embed.unsqueeze(1).expand(-1, A, -1)
        tiled_obs_embed = tiled_obs_embed / ((((tiled_obs_embed ** 2).sum(dim=-1, keepdim=True)) ** 0.5).detach())

        raw_logits = (tiled_obs_embed * action_embed).sum(dim=-1)
        raw_logits = raw_logits.masked_fill((1.0 - padded_masks).bool(), float('-inf'))
        raw_logits = raw_logits.masked_fill((1.0 - action_masks).bool(), float('-inf'))

        pi = F.softmax(raw_logits * self.prior_beta, dim=-1)

        epsilon_action_num = action_masks.sum(dim=-1, keepdim=True).float()
        random_pi = torch.ones_like(pi)
        random_pi = random_pi / epsilon_action_num
        eps = 0.05
        pi = (1 - eps) * pi + random_pi * eps


        return pi

    def action_scoring(self, action_candidate_list, h_og=None, obs_mask=None, h_go=None, node_mask=None, previous_h=None, previous_c=None, use_model=None):
        model = self.choose_model(use_model)
        # step 4
        input_action_candidate = self.get_action_candidate_list_input(action_candidate_list)
        action_scores, action_logits, action_masks, new_h, new_c = model.score_actions(input_action_candidate, h_og, obs_mask, h_go, node_mask, previous_h, previous_c)  # batch x num_actions

        pi = torch.nn.functional.softmax(action_logits, dim=-1)

        epsilon_action_num = action_masks.sum(dim=-1, keepdim=True).float()
        random_pi = torch.ones_like(pi)
        random_pi = random_pi / epsilon_action_num
        eps = 0.05
        pi = (1 - eps) * pi + random_pi * eps


        return action_scores, pi, action_masks, new_h, new_c

    # action scoring stuff (Deep Q-Learning)
    def choose_random_action(self, action_rank, action_unpadded=None):
        """
        Select an action randomly.
        """
        batch_size = action_rank.size(0)
        action_space_size = action_rank.size(1)
        if action_unpadded is None:
            indices = np.random.choice(action_space_size, batch_size)
        else:
            indices = []
            for j in range(batch_size):
                indices.append(np.random.choice(len(action_unpadded[j])))
            indices = np.array(indices)
        return indices

    def choose_sampled_action(self, action_rank, action_mask=None):
        """
        Generate an action by maximum q values.
        """
        #action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2  # minus the min value, so that all values are non-negative
        if action_mask is not None:
            assert action_mask.size() == action_rank.size(), (action_mask.size().shape, action_rank.size())
            action_probs = action_rank * action_mask
        action_dist = Categorical(probs=action_probs)
        #action_indices = torch.argmax(action_rank, -1)  # batch
        action_indices = action_dist.sample() # batch

        return to_np(action_indices)

    def choose_maxQ_action(self, action_rank, action_mask=None):
        """
        Generate an action by maximum q values.
        """
        action_rank = action_rank - torch.min(action_rank, -1, keepdim=True)[0] + 1e-2  # minus the min value, so that all values are non-negative
        if action_mask is not None:
            assert action_mask.size() == action_rank.size(), (action_mask.size().shape, action_rank.size())
            action_rank = action_rank * action_mask
        action_indices = torch.argmax(action_rank, -1)  # batch
        return to_np(action_indices)

    def act_greedy(self, observation_strings, graph_input, action_candidate_list, previous_h=None, previous_c=None):
        with torch.no_grad():
            h_og, obs_mask, h_go, node_mask = self.encode(observation_strings, graph_input, use_model="online")
            action_scores, _, action_masks, new_h, new_c = self.action_scoring(action_candidate_list, h_og, obs_mask, h_go, node_mask, previous_h, previous_c, use_model="online")
            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            chosen_indices = action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, new_h, new_c

    def act_random(self, observation_strings, graph_input, action_candidate_list, previous_h=None, previous_c=None):
        with torch.no_grad():
            h_og, obs_mask, h_go, node_mask = self.encode(observation_strings, graph_input, use_model="online")
            action_scores, _, _, new_h, new_c = self.action_scoring(action_candidate_list, h_og, obs_mask, h_go, node_mask, previous_h, previous_c, use_model="online")
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            chosen_indices = action_indices_random
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            return chosen_actions, chosen_indices, new_h, new_c

    def act(self, observation_strings, graph_input, action_candidate_list, previous_h=None, previous_c=None, random=False):

        with torch.no_grad():
            if self.mode == "eval":
                return self.act_greedy(observation_strings, graph_input, action_candidate_list, previous_h, previous_c)
            if random:
                return self.act_random(observation_strings, graph_input, action_candidate_list, previous_h, previous_c)
            batch_size = len(observation_strings)

            h_og, obs_mask, h_go, node_mask = self.encode(observation_strings, graph_input, use_model="online")
            action_scores, action_probs, action_masks, new_h, new_c = self.action_scoring(action_candidate_list, h_og, obs_mask, h_go, node_mask, previous_h, previous_c, use_model="online")

            #action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            action_indices_sampled = self.choose_sampled_action(action_probs, action_masks)
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            # random number for epsilon greedy
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            less_than_epsilon = (rand_num < self.epsilon).astype("float32")  # batch
            greater_than_epsilon = 1.0 - less_than_epsilon

            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_sampled
            chosen_indices = chosen_indices.astype(int)
            try:
                chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, chosen_indices)]
            except:
                for item, idx in zip(action_candidate_list, chosen_indices):
                    print(item, len(item), idx)

            return chosen_actions, chosen_indices, new_h, new_c

    def update_critic(self, data):

        obs_list, _, candidate_list, action_indices, graph_triplet_list, rewards, next_obs_list, _, next_candidate_list, next_graph_triplet_list, actual_indices, actual_ns, prior_weights = data

        h_og, obs_mask, h_go, node_mask = self.encode(obs_list, graph_triplet_list, use_model="online")
        action_scores, _, _, _, _ = self.action_scoring(candidate_list, h_og, obs_mask, h_go, node_mask, None, None, use_model="online")

        # ps_a
        action_indices = to_pt(action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)
        q_value = ez_gather_dim_1(action_scores, action_indices).squeeze(1)  # batch

        with torch.no_grad():
            if self.noisy_net:
                self.target_net.reset_noise()  # Sample new target net noise
            # pns Probabilities p(s_t+n, ·; θonline)
            h_og, obs_mask, h_go, node_mask = self.encode(next_obs_list, next_graph_triplet_list, use_model="online")
            _, next_action_probs, next_action_masks, _, _ = self.action_scoring(next_candidate_list, h_og, obs_mask, h_go, node_mask, None, None, use_model="online")

            next_action_indices = self.choose_sampled_action(next_action_probs, next_action_masks)  # batch
            next_action_indices = to_pt(next_action_indices, enable_cuda=self.use_cuda, type='long').unsqueeze(-1)

            next_action_probs = next_action_probs.masked_fill((1.0 - next_action_masks).bool(), 1)
            next_action_logits = next_action_probs.log()
            next_action_logits = next_action_logits.masked_fill((1.0 - next_action_masks).bool(), 0)
            
            next_action_prior_probs = self.get_prior_probs(next_obs_list, next_candidate_list, next_action_masks, use_target=True)
            next_action_prior_probs = next_action_prior_probs.masked_fill((1.0 - next_action_masks).bool(), 1)
            next_action_prior_logits = next_action_prior_probs.log()
            next_action_prior_logits = next_action_prior_logits.masked_fill((1.0 - next_action_masks).bool(), 0)

            # pns # Probabilities p(s_t+n, ·; θtarget)
            h_og, obs_mask, h_go, node_mask = self.encode(next_obs_list, next_graph_triplet_list, use_model="target")
            next_action_scores, _, _, _, _ = self.action_scoring(next_candidate_list, h_og, obs_mask, h_go, node_mask, None, None, use_model="target")

            # pns_a # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            next_q_value = ez_gather_dim_1(next_action_scores, next_action_indices).squeeze(1)  # batch
            next_logit = ez_gather_dim_1(next_action_logits, next_action_indices).squeeze(1)  # batch
            next_prior_logit = ez_gather_dim_1(next_action_prior_logits, next_action_indices).squeeze(1) # batch
            discount = to_pt((np.ones_like(actual_ns) * self.discount_gamma_game_reward) ** actual_ns, self.use_cuda, type="float")


        rewards = rewards + (next_q_value - self.alpha * (next_logit - next_prior_logit)) * discount  # batch
        critic_loss = F.smooth_l1_loss(q_value, rewards, reduce=False)  # batch

        #prior_weights = to_pt(prior_weights, enable_cuda=self.use_cuda, type="float")
        #critic_loss = critic_loss * prior_weights
        critic_loss = critic_loss
        critic_loss = torch.mean(critic_loss)

        abs_td_error = np.abs(to_np(q_value - rewards))
        new_priorities = abs_td_error + self.prioritized_replay_eps
        self.dqn_memory.update_priorities(actual_indices, new_priorities)

        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        critic_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients

        return critic_loss, q_value

    def update_policy(self, data):

        obs_list, _, candidate_list, action_indices, graph_triplet_list, rewards, next_obs_list, _, next_candidate_list, next_graph_triplet_list, actual_indices, actual_ns, prior_weights = data

        h_og, obs_mask, h_go, node_mask = self.encode(obs_list, graph_triplet_list, use_model="online")
        action_scores, action_probs, action_masks, _, _ = self.action_scoring(candidate_list, h_og, obs_mask, h_go, node_mask, None, None, use_model="online")
        action_prior_probs = self.get_prior_probs(obs_list, candidate_list, action_masks, use_target=True)

        action_scores = action_scores.detach()
        action_prior_probs = action_prior_probs.detach()

        action_probs = action_probs.masked_fill((1.0 - action_masks).bool(), 1)
        action_logits = action_probs.log()

        action_prior_probs = action_prior_probs.masked_fill((1.0 - action_masks).bool(), 1)
        action_prior_logits = action_prior_probs.log()

        pi = action_probs

        pi = pi.masked_fill((1.0 - action_masks).bool(), 0)
        action_logits = action_logits.masked_fill((1.0 - action_masks).bool(), 0)
        action_prior_logits = action_prior_logits.masked_fill((1.0 - action_masks).bool(), 0)
        action_scores = action_scores.masked_fill((1.0 - action_masks).bool(), 0)


        policy_loss = pi * (self.alpha * (action_logits - action_prior_logits) - action_scores)
        policy_loss = policy_loss.masked_fill((1.0 - action_masks).bool(), 0)
        policy_loss = policy_loss.sum(dim=-1)
        policy_loss = torch.mean(policy_loss)

        # Backpropagate
        self.online_net.zero_grad()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.policy_optimizer.step()  # apply gradients

        return policy_loss

    def update_prior(self, data):

        obs_list, _, candidate_list, action_indices, graph_triplet_list, rewards, next_obs_list, _, next_candidate_list, next_graph_triplet_list, actual_indices, actual_ns, prior_weights = data

        h_og, obs_mask, h_go, node_mask = self.encode(obs_list, graph_triplet_list, use_model="online")
        _, action_probs, action_masks, _, _ = self.action_scoring(candidate_list, h_og, obs_mask, h_go, node_mask, None, None, use_model="online")
        action_prior_probs = self.get_prior_probs(obs_list, candidate_list, action_masks, use_target=False)

        action_probs = action_probs.detach()

        action_probs = action_probs.masked_fill((1.0 - action_masks).bool(), 1)
        action_logits = action_probs.log()

        action_prior_probs = action_prior_probs.masked_fill((1.0 - action_masks).bool(), 1)
        action_prior_logits = action_prior_probs.log()

        pi = action_probs

        pi = pi.masked_fill((1.0 - action_masks).bool(), 0)
        action_logits = action_logits.masked_fill((1.0 - action_masks).bool(), 0)
        action_prior_logits = action_prior_logits.masked_fill((1.0 - action_masks).bool(), 0)

        prior_loss = - pi * (action_prior_logits)
        prior_loss = prior_loss.masked_fill((1.0 - action_masks).bool(), 0)
        prior_loss = prior_loss.sum(dim=-1)
        prior_loss = torch.mean(prior_loss)

        self.prior.zero_grad()
        self.prior_optimizer.zero_grad()
        prior_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.prior.parameters(), self.clip_grad_norm)
        self.prior_optimizer.step()

        return prior_loss

    def get_ac_loss(self, episode_no):
        """
        Update neural model in agent. In this example we follow algorithm
        of updating model in dqn with replay memory.
        """
        if len(self.dqn_memory) < self.replay_batch_size:
            return None, None, None, None

        data = self.dqn_memory.sample(self.replay_batch_size, beta=self.beta_scheduler.value(episode_no), multi_step=self.multi_step)
        prior_data = self.prior_memory.sample(self.replay_batch_size, beta=self.beta_scheduler.value(episode_no), multi_step=self.multi_step)

        if data is None:
            return None, None, None, None

        critic_loss, q_value = self.update_critic(data)
        policy_loss = self.update_policy(data)
        if self.use_prior:
            prior_loss = self.update_prior(prior_data)
        else:
            prior_loss = policy_loss * 0

        return critic_loss, policy_loss, prior_loss, q_value


    def update_model(self, episode_no):
        # update neural model by replaying snapshots in replay memory
        critic_loss, policy_loss, prior_loss, q_value = self.get_ac_loss(episode_no)
        print("critic_loss : {}, policy_loss: {}, prior_loss: {}".format(critic_loss, policy_loss, prior_loss))
        return to_np(torch.mean(critic_loss)), to_np(torch.mean(q_value))

    def update_prior_only(self, episode_no):
        if len(self.prior_memory) < self.replay_batch_size:
            return None
        prior_data = self.prior_memory.sample(self.replay_batch_size, beta=self.beta_scheduler.value(episode_no), multi_step=self.multi_step)

        if self.use_prior:
            prior_loss = self.update_prior(prior_data)
        else:
            prior_loss = 0

        print("prior_loss: {}".format(prior_loss))

        return prior_loss


    def get_graph_rewards(self, prev_triplets, current_triplets):
        batch_size = len(current_triplets)
        if self.graph_reward_lambda == 0:
            return [0.0 for _ in current_triplets]

        if self.graph_reward_type == "triplets_increased":
            rewards = [float(len(c_triplet) - len(p_triplet)) for p_triplet, c_triplet in zip(prev_triplets, current_triplets)]
        elif self.graph_reward_type == "triplets_difference":
            rewards = []
            for b in range(batch_size):
                curr = current_triplets[b]
                prev = prev_triplets[b]
                curr = set(["|".join(item) for item in curr])
                prev = set(["|".join(item) for item in prev])
                diff_num = len(prev - curr) + len(curr - prev)
                rewards.append(float(diff_num))
        else:
            raise NotImplementedError
        rewards = [min(1.0, max(0.0, float(item) * self.graph_reward_lambda)) for item in rewards]
        return rewards

    def reset_binarized_counter(self, batch_size):
        self.binarized_counter_dict = [{} for _ in range(batch_size)]

    def get_binarized_count(self, observation_strings, update=True):
        batch_size = len(observation_strings)
        count_rewards = []
        for i in range(batch_size):
            concat_string = observation_strings[i]
            if concat_string not in self.binarized_counter_dict[i]:
                self.binarized_counter_dict[i][concat_string] = 0.0
            if update:
                self.binarized_counter_dict[i][concat_string] += 1.0
            r = self.binarized_counter_dict[i][concat_string]
            r = float(r == 1.0)
            count_rewards.append(r)
        return count_rewards
