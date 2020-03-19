import difflib


import overrides
import sqlparse
import torch
from typing import Dict, Any, List, Tuple, Mapping, Sequence
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule, ProductionRuleArray
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed, Attention, Embedding, \
    FeedForward
from allennlp.nn import util, Activation
from allennlp.state_machines.states import RnnStatelet, GrammarStatelet
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.state_machines.transition_functions import LinkingTransitionFunction

from dataset_reader.dataset_util.utils import action_sequence_to_sql
from modules.memory_schema_attn import MemAttn
from semparse.worlds.spider_world import SpiderWorld
from state_machines.states.grammar_based_state import GrammarBasedState
from state_machines.states.sql_state import SqlState
from state_machines.transition_functions.attend_past_schema_items_transition import \
    AttendPastSchemaItemsTransitionFunction


@Model.register("MMParser")
class MMParser(Model):
    # Seq2Seq encoder with Memory Network
    #
    #  Architecture
    #   1) RNN encoder for input symbols in query and input memory items (either shared or separate)
    #   2) RNN encoder for output symbols in the output memory items
    #   3) Key-value memory for embedding query items with support context
    #   4) State machine decoder for output symbols based on SQL grammar
    def __init__(self,
                 question_embedder: TextFieldEmbedder,
                 input_memory_embedder: TextFieldEmbedder,
                 output_memory_embedder: TextFieldEmbedder,
                 question_encoder: Seq2SeqEncoder,
                 input_memory_encoder: Seq2VecEncoder,
                 output_memory_encoder: Seq2VecEncoder,
                 decoder_beam_search: BeamSearch,
                 input_attention: Attention,
                 past_attention: Attention,
                 action_embedding_dim: int,
                 max_decoding_steps: int,
                 nhop: int,
                 decoding_nhop: int,
                 vocab: Vocabulary,
                 dataset_path: str = 'dataset',
                 parse_sql_on_decoding: bool = True,
                 training_beam_size: int = None,
                 add_action_bias: bool = True,
                 decoder_self_attend: bool = True,
                 decoder_num_layers: int = 1,
                 dropout: float = 0.0,
                 rule_namespace: str = 'rule_labels') -> None:
        super().__init__(vocab)

        self.question_embedder = question_embedder
        self._input_mm_embedder = input_memory_embedder
        self._output_mm_embedder = output_memory_embedder
        self._question_encoder = question_encoder
        self._input_mm_encoder = TimeDistributed(input_memory_encoder)
        self._output_mm_encoder = TimeDistributed(output_memory_encoder)
        
        self.parse_sql_on_decoding = parse_sql_on_decoding
        self._self_attend = decoder_self_attend
        self._max_decoding_steps = max_decoding_steps
        self._add_action_bias = add_action_bias
        self._rule_namespace = rule_namespace
        num_actions = vocab.get_vocab_size(self._rule_namespace)
        if self._add_action_bias:
            input_action_dim = action_embedding_dim + 1
        else:
            input_action_dim = action_embedding_dim
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=input_action_dim)
        self._input_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)
        self._output_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)

        self._num_entity_types = 9
        self._entity_type_decoder_input_embedding= Embedding(self._num_entity_types, action_embedding_dim)
        self._entity_type_decoder_output_embedding = Embedding(self._num_entity_types, action_embedding_dim)

        self._entity_type_encoder_embedding = Embedding(self._num_entity_types, (int)(question_encoder.get_output_dim()/2))

        self._decoder_num_layers = decoder_num_layers
        self._action_embedding_dim = action_embedding_dim

        self._ent2ent_ff = FeedForward(action_embedding_dim, 1, action_embedding_dim, Activation.by_name('relu')())

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_utterance = torch.nn.Parameter(
            torch.FloatTensor(question_encoder.get_output_dim()))
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_utterance)

        if self._self_attend:
            self._transition_function = AttendPastSchemaItemsTransitionFunction(
                encoder_output_dim=question_encoder.get_output_dim(),
                action_embedding_dim=action_embedding_dim,
                input_attention=input_attention,
                past_attention=past_attention,
                decoding_nhop=decoding_nhop,
                predict_start_type_separately=False,
                add_action_bias=self._add_action_bias,
                dropout=dropout,
                num_layers=self._decoder_num_layers)
        else:
            self._transition_function = LinkingTransitionFunction(encoder_output_dim=question_encoder.get_output_dim(),
                                                                  action_embedding_dim=action_embedding_dim,
                                                                  input_attention=input_attention,
                                                                  predict_start_type_separately=False,
                                                                  add_action_bias=self._add_action_bias,
                                                                  dropout=dropout,
                                                                  num_layers=self._decoder_num_layers)

        self._mm_attn = MemAttn(question_encoder.get_output_dim(), nhop)

        self._beam_search = decoder_beam_search
        self._decoder_trainer = MaximumMarginalLikelihood(training_beam_size)
        
        self._action_padding_index = -1  # the padding value used by IndexField

        self._exact_match = Average()
        self._sql_evaluator_match = Average()
        self._action_similarity = Average()
        self._acc_single = Average()
        self._acc_multi = Average()
        self._beam_hit = Average()

        # TODO: Remove hard-coded dirs
        self._evaluate_func = partial(evaluate,
                                      db_dir=os.path.join(dataset_path, 'database'),
                                      table=os.path.join(dataset_path, 'tables.json'),
                                      check_valid=False)

    @overrides
    def forward(self,
                utterance: Dict[str, torch.LongTensor],
                valid_actions: List[List[ProductionRule]],
                world: List[SpiderWorld],
                schema: Dict[str, torch.LongTensor],
                action_sequence: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        device = utterance['tokens'].device
        initial_state = self._get_initial_state(utterance, world, schema, valid_actions)

        if action_sequence is not None:
            # Remove the trailing dimension (from ListField[ListField[IndexField]]).
            action_sequence = action_sequence.squeeze(-1)
            action_mask = action_sequence != self._action_padding_index
        else:
            action_mask = None

        if self.training:
            decode_output = self._decoder_trainer.decode(initial_state,
                                                         self._transition_function,
                                                         (action_sequence.unsqueeze(1), action_mask.unsqueeze(1)))

            return {'loss': decode_output['loss']}
        else:
            loss = torch.tensor([0]).float().to(device)
            if action_sequence is not None and action_sequence.size(1) > 1:
                try:
                    loss = self._decoder_trainer.decode(initial_state,
                                                        self._transition_function,
                                                        (action_sequence.unsqueeze(1),
                                                         action_mask.unsqueeze(1)))['loss']
                except ZeroDivisionError:
                    # reached a dead-end during beam search
                    pass

            outputs: Dict[str, Any] = {
                'loss': loss
            }

            num_steps = self._max_decoding_steps
            # This tells the state to start keeping track of debug info, which we'll pass along in
            # our output dictionary.
            # initial_state.debug_info = [[] for _ in range(batch_size)]

            best_final_states = self._beam_search.search(num_steps,
                                                         initial_state,
                                                         self._transition_function,
                                                         keep_final_unfinished_states=False)

            self._compute_validation_outputs(valid_actions,
                                             best_final_states,
                                             world,
                                             action_sequence,
                                             outputs)
        return outputs

    def _get_initial_state(self,
                           utterance: Dict[str, torch.LongTensor],
                           worlds: List[SpiderWorld],
                           schema: Dict[str, torch.LongTensor],
                           valid_actions: List[List[ProductionRule]]) -> GrammarBasedState:

        utterance_mask = util.get_text_field_mask(utterance).float()
        embedded_utterance = self.question_embedder(utterance)
        batch_size, _, _ = embedded_utterance.size()
        encoder_outputs = self._dropout(self._question_encoder(embedded_utterance, utterance_mask))

        schema_text = schema['text']
        input_mm_schema = self._input_mm_embedder(schema_text, num_wrapping_dims=1)
        output_mm_schema = self._output_mm_embedder(schema_text, num_wrapping_dims=1)
        batch_size, num_entities, num_entity_tokens, _ = input_mm_schema.size()
        schema_mask = util.get_text_field_mask(schema_text, num_wrapping_dims=1).float()

        # TODO
        # entity_types: tensor with shape (batch_size, num_entities), where each entry is the
        # entity's type id.
        # entity_type_dict: Dict[int, int], mapping flattened_entity_index -> type_index
        # These encode the same information, but for efficiency reasons later it's nice
        # to have one version as a tensor and one that's accessible on the cpu.
        entity_types, entity_type_dict = self._get_type_vector(worlds, num_entities, input_mm_schema.device)
        # (batch_size, num_entities, embedding_dim)
        entity_type_embeddings = self._entity_type_encoder_embedding(entity_types)

        # (batch_size, num_entities, embedding_dim)
        # An entity memory-representation is concatenated with two parts:
        # 1. Entity tokens embedding
        # 2. Entity type embedding
        K = torch.cat([self._input_mm_encoder(input_mm_schema, schema_mask),
                       entity_type_embeddings], dim = 2)
        V = torch.cat([self._output_mm_encoder(output_mm_schema, schema_mask),
                       entity_type_embeddings], dim = 2)
        encoder_output_dim = self._question_encoder.get_output_dim()

        # Encodes utterance in the context of the schema, which is stored in external memory
        encoder_outputs_with_context, attn_weights = self._mm_attn(encoder_outputs, K, V)
        attn_weights = attn_weights.transpose(1,2)
        final_encoder_output = util.get_final_encoder_states(encoder_outputs_with_context,
                                                             utterance_mask,
                                                             self._question_encoder.is_bidirectional())

        max_entities_relevance = attn_weights.max(dim=2)[0]
        entities_relevance = max_entities_relevance.unsqueeze(-1).detach()
        if self._self_attend:
            entities_ff = self._ent2ent_ff(entity_type_embeddings * entities_relevance)
            linked_actions_linking_scores = torch.bmm(entities_ff, entities_ff.transpose(1, 2))
        else:
            linked_actions_linking_scores = [None] * batch_size

        memory_cell = encoder_outputs.new_zeros(batch_size, encoder_output_dim)
        initial_score = embedded_utterance.data.new_zeros(batch_size)

        # To make grouping states together in the decoder easier, we convert the batch dimension in
        # all of our tensors into an outer list.  For instance, the encoder outputs have shape
        # `(batch_size, utterance_length, encoder_output_dim)`.  We need to convert this into a list
        # of `batch_size` tensors, each of shape `(utterance_length, encoder_output_dim)`.  Then we
        # won't have to do any index selects, or anything, we'll just do some `torch.cat()`s.
        initial_score_list = [initial_score[i] for i in range(batch_size)]
        encoder_output_list = [encoder_outputs[i] for i in range(batch_size)]
        utterance_mask_list = [utterance_mask[i] for i in range(batch_size)]
        # RnnStatelet is using to keep track of the internal state of a decoder RNN:
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
                                                 memory_cell[i],
                                                 self._first_action_embedding,
                                                 self._first_attended_utterance,
                                                 encoder_output_list,
                                                 utterance_mask_list))

        initial_grammar_state = [self._create_grammar_state(worlds[i],
                                                            valid_actions[i],
                                                            attn_weights[i],
                                                            linked_actions_linking_scores[i],
                                                            entity_types[i])
                                 for i in range(batch_size)]

        initial_sql_state = [SqlState(valid_actions[i], self.parse_sql_on_decoding) for i
                             in
                             range(batch_size)]

        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          sql_state=initial_sql_state,
                                          possible_actions=valid_actions,
                                          action_entity_mapping=[w.get_action_entity_mapping() for w in worlds])

        return initial_state

    @staticmethod
    def _get_type_vector(worlds: List[SpiderWorld],
                         num_entities: int,
                         device) -> Tuple[torch.LongTensor, Dict[int, int]]:
        """
        Produces the encoding for each entity's type. In addition, a map from a flattened entity
        index to type is returned to combine entity type operations into one method.

        Parameters
        ----------
        worlds : ``List[AtisWorld]``
        num_entities : ``int``
        tensor : ``torch.Tensor``
            Used for copying the constructed list onto the right device.

        Returns
        -------
        A ``torch.LongTensor`` with shape ``(batch_size, num_entities, num_types)``.
        entity_types : ``Dict[int, int]``
            This is a mapping from ((batch_index * num_entities) + entity_index) to entity type id.
        """
        entity_types = {}
        batch_types = []

        column_type_ids = ['boolean', 'foreign', 'number', 'others', 'primary', 'text', 'time']

        for batch_index, world in enumerate(worlds):
            types = []

            for entity_index, entity in enumerate(world.db_context.knowledge_graph.entities):
                parts = entity.split(':')
                entity_main_type = parts[0]
                if entity_main_type == 'column':
                    column_type = parts[1]
                    entity_type = column_type_ids.index(column_type)
                elif entity_main_type == 'string':
                    # cell value
                    entity_type = len(column_type_ids)
                elif entity_main_type == 'table':
                    entity_type = len(column_type_ids) + 1
                else:
                    raise (Exception("Unkown entity"))
                types.append(entity_type)

                # For easier lookups later, we're actually using a _flattened_ version
                # of (batch_index, entity_index) for the key, because this is how the
                # linking scores are stored.
                flattened_entity_index = batch_index * num_entities + entity_index
                entity_types[flattened_entity_index] = entity_type
            padded = pad_sequence_to_length(types, num_entities, lambda: 0)
            batch_types.append(padded)

        return torch.tensor(batch_types, dtype=torch.long, device=device), entity_types

    def _create_grammar_state(self,
                              world: SpiderWorld,
                              possible_actions: List[ProductionRule],
                              attn_weights: torch.Tensor,
                              linked_actions_linking_scores: torch.Tensor,
                              entity_types: torch.Tensor) -> GrammarStatelet:
        action_map = {}
        for action_index, action in enumerate(possible_actions):
            action_string = action[0]
            action_map[action_string] = action_index

        valid_actions = world.valid_actions
        entity_map = {}
        entities = world.entities_names
        for entity_index, entity in enumerate(entities):
            entity_map[entity] = entity_index
        
        translated_valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.  We'll first split those productions by global vs.
            # linked action.

            action_indices = [action_map[action_string] for action_string in action_strings]
            production_rule_arrays = [(possible_actions[index], index) for index in action_indices]
            global_actions = []
            linked_actions = []
            for production_rule_array, action_index in production_rule_arrays:
                if production_rule_array[1]:
                    global_actions.append((production_rule_array[2], action_index))
                else:
                    linked_actions.append((production_rule_array[0], action_index))

            if global_actions:
                global_action_tensors, global_action_ids = zip(*global_actions)
                global_action_tensor = torch.cat(global_action_tensors, dim=0).to(
                    global_action_tensors[0].device).long()
                global_input_embeddings = self._action_embedder(global_action_tensor)
                global_input_action_embeddings = self._input_action_embedder(global_action_tensor)
                global_output_action_embeddings = self._output_action_embedder(global_action_tensor)
                translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                           global_input_action_embeddings,
                                                           list(global_action_ids),
                                                           global_output_action_embeddings)

            if linked_actions:
                linked_rules, linked_action_ids = zip(*linked_actions)
                entities = [rule.split(' -> ')[1].strip('[]\"') for rule in linked_rules]
                entity_ids = [entity_map[entity] for entity in entities]
                
                entity_linking_scores = attn_weights[entity_ids]
                if linked_actions_linking_scores is not None:
                    entity_action_linking_scores = linked_actions_linking_scores[entity_ids]
                # if not self._decoder_use_graph_entities:
                entity_type_tensor = entity_types[entity_ids]
                entity_type_input_embeddings = (self._entity_type_decoder_input_embedding(entity_type_tensor)
                                          .to(entity_types.device)
                                          .float())
                entity_type_output_embeddings = (self._entity_type_decoder_output_embedding(entity_type_tensor)
                                          .to(entity_types.device)
                                          .float())
                #entity_type_input_embeddings = None
                #entity_type_output_embeddings = None
                # else:
                #     entity_type_embeddings = entity_graph_encoding.index_select(
                #         dim=0,
                #         index=torch.tensor(entity_ids, device=entity_graph_encoding.device)
                #     )

                if self._self_attend:
                    translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                               entity_type_input_embeddings,
                                                               list(linked_action_ids),
                                                               entity_action_linking_scores,
                                                               entity_type_output_embeddings)
                else:
                    translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                               entity_type_input_embeddings,
                                                               list(linked_action_ids),
                                                               entity_type_output_embeddings)

        return GrammarStatelet(['statement'],
                               translated_valid_actions,
                               self.is_nonterminal)

    @staticmethod
    def is_nonterminal(token: str):
        if token[0] == '"' and token[-1] == '"':
            return False
        return True

    @staticmethod
    def _action_history_match(predicted: List[int], targets: torch.LongTensor) -> int:
        # TODO(mattg): this could probably be moved into a FullSequenceMatch metric, or something.
        # Check if target is big enough to cover prediction (including start/end symbols)
        if len(predicted) > targets.size(0):
            return 0
        predicted_tensor = targets.new_tensor(predicted)
        targets_trimmed = targets[:len(predicted)]
        # Return 1 if the predicted sequence is anywhere in the list of targets.
        return torch.max(torch.min(targets_trimmed.eq(predicted_tensor), dim=0)[0]).item()

    @staticmethod
    def _query_difficulty(targets: torch.LongTensor, action_mapping, batch_index):
        number_tables = len([action_mapping[(batch_index, int(a))] for a in targets if
                             a >= 0 and action_mapping[(batch_index, int(a))].startswith('table_name')])
        return number_tables > 1
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            '_match/exact_match': self._exact_match.get_metric(reset),
            'sql_match': self._sql_evaluator_match.get_metric(reset),
            '_others/action_similarity': self._action_similarity.get_metric(reset),
            '_match/match_single': self._acc_single.get_metric(reset),
            '_match/match_hard': self._acc_multi.get_metric(reset),
            'beam_hit': self._beam_hit.get_metric(reset)
        }

    def find_shortest_path(self, start, end, graph):
        stack = [[start, []]]
        visited = set()
        while len(stack) > 0:
            ele, history = stack.pop()
            if ele == end:
                return history
            for node in graph[ele]:
                if node[0] not in visited:
                    stack.append((node[0], history + [(node[0], node[1])]))
                    visited.add(node[0])
        # print("table {} table {}".format(start,end))

    def _add_from_clause(self, origin_query, world: SpiderWorld):
        predicted_sql_query_tokens = origin_query.split(" ")
        # print("predicted_sql_query_tokens:{}".format(predicted_sql_query_tokens))

        select_indices = [i for i, x in enumerate(predicted_sql_query_tokens) if x == "select"]
        select_indices.append(len(predicted_sql_query_tokens))
        # From bottom to top
        select_indices.reverse()
        dbs_json_blob = json.load(open(world.db_context.tables_file, "r"))
        graph = defaultdict(list)
        table_list = []
        dbtable = {}
        for table in dbs_json_blob:
            if world.db_id == table['db_id']:
                dbtable = table
                for acol, bcol in table["foreign_keys"]:
                    t1 = table["column_names"][acol][0]
                    t2 = table["column_names"][bcol][0]
                    graph[t1].append((t2, (acol, bcol)))
                    graph[t2].append((t1, (bcol, acol)))
                table_list = [table for table in table["table_names_original"]]
        # print("table_list:{}".format(table_list))

        end_idx = select_indices[0]
        for index in select_indices[1:]:
            table_alias_dict = {}
            idx = 1

            start_idx = index
            tables = set(
                [token.split(".")[0] for token in predicted_sql_query_tokens[start_idx: end_idx] if '.' in token])
            # print(tables)
            candidate_tables: List[int] = []
            for table in tables:
                for i, table1 in enumerate(table_list):
                    if table1.lower() == table:
                        candidate_tables.append(i)
                        break
            # print("\ncandidate_tables:{}".format(candidate_tables))
            ret = ""
            flag_only_sel_count = False
            if len(candidate_tables) > 1:
                start = candidate_tables[0]
                table_alias_dict[start] = idx
                idx += 1
                ret = "from {}".format(dbtable["table_names_original"][start])
                try:
                    for end in candidate_tables[1:]:
                        if end in table_alias_dict:
                            continue
                        path = self.find_shortest_path(start, end, graph)
                        # print("got path = {}".format(path))
                        prev_table = start
                        if not path:
                            table_alias_dict[end] = idx
                            idx += 1
                            ret = "{} join {}".format(ret, dbtable["table_names_original"][end])
                            continue
                        for node, (acol, bcol) in path:
                            if node in table_alias_dict:
                                prev_table = node
                                continue
                            table_alias_dict[node] = idx
                            idx += 1
                            # print("test every slot:")
                            # print("table:{}, dbtable:{}".format(table, dbtable))
                            # print(dbtable["table_names_original"][node])
                            # print(dbtable["table_names_original"][prev_table])
                            # print(dbtable["column_names_original"][acol][1])
                            # print(dbtable["table_names_original"][node])
                            # print(dbtable["column_names_original"][bcol][1])
                            ret = "{} join {} on {}.{} = {}.{}".format(ret, dbtable["table_names_original"][node],
                                                                       dbtable["table_names_original"][prev_table],
                                                                       dbtable["column_names_original"][acol][1],
                                                                       dbtable["table_names_original"][node],
                                                                       dbtable["column_names_original"][bcol][1])
                            prev_table = node

                except:
                    print("\n!!Exception in spider_parser.py : line 924!! \npredicted_sql_query_tokens:{}".format(
                        predicted_sql_query_tokens))
                # print(ret)
            # If all the columns from one table, generate FROM Clause directly
            elif len(candidate_tables) == 1:
                ret = "from {}".format(tables.pop())
                # print("\nret:{}".format(ret))
            else:
                ret = 'from'
                flag_only_sel_count = True

            if not flag_only_sel_count:
                flag = False
                index = start_idx + len(predicted_sql_query_tokens[start_idx:end_idx])
                brace_count = 0
                for ii, token in enumerate(predicted_sql_query_tokens[start_idx:end_idx]):
                    if token == "(":
                        brace_count += 1
                    if token == ")":
                        if brace_count == 0:
                            index = ii + start_idx
                            predicted_sql_query_tokens = predicted_sql_query_tokens[:index] + [ret] + \
                                                         predicted_sql_query_tokens[index:]
                            # print(predicted_sql_query_tokens)
                            flag = True
                            break
                        else:
                            brace_count -= 1
                    if token == "where" or token == "group" or token == "order":
                        index = ii + start_idx
                        predicted_sql_query_tokens = predicted_sql_query_tokens[:index] + [ret] + \
                                                     predicted_sql_query_tokens[index:]
                        flag = True
                        # print(predicted_sql_query_tokens)
                        break
                if not flag:
                    predicted_sql_query_tokens = predicted_sql_query_tokens[:index] + [ret]
                    # print("\npredicted_sql_query_tokens:{}".format(' '.join([token for token in predicted_sql_query_tokens])))
            else:
                for ii, token in enumerate(predicted_sql_query_tokens[start_idx:end_idx]):
                    if token == "from_count":
                        predicted_sql_query_tokens = predicted_sql_query_tokens[:ii] + [ret] + \
                                                     predicted_sql_query_tokens[ii + 1:]
                        # print("\npredicted_sql_query:{}".format(' '.join([token for token in predicted_sql_query_tokens])))
                        break
            end_idx = start_idx
            # print("predicted_sql_query_tokens:{}".format(predicted_sql_query_tokens))
        return ' '.join(['*' if '.*' in token else token for token in predicted_sql_query_tokens])

    def _compute_validation_outputs(self,
                                    actions: List[List[ProductionRuleArray]],
                                    best_final_states: Mapping[int, Sequence[GrammarBasedState]],
                                    world: List[SpiderWorld],
                                    target_list: List[List[str]],
                                    outputs: Dict[str, Any]) -> None:
        batch_size = len(actions)

        outputs['predicted_sql_query'] = []

        action_mapping = {}
        for batch_index, batch_actions in enumerate(actions):
            for action_index, action in enumerate(batch_actions):
                action_mapping[(batch_index, action_index)] = action[0]

        for i in range(batch_size):
            # gold sql exactly as given
            original_gold_sql_query = ' '.join(world[i].get_query_without_table_hints())

            if i not in best_final_states:
                self._exact_match(0)
                self._action_similarity(0)
                self._sql_evaluator_match(0)
                self._acc_multi(0)
                self._acc_single(0)
                outputs['predicted_sql_query'].append('')
                continue

            best_action_indices = best_final_states[i][0].action_history[0]

            action_strings = [action_mapping[(i, action_index)]
                              for action_index in best_action_indices]
            predicted_sql_query = action_sequence_to_sql(action_strings, add_table_names=True)
            # print ("predicted_sql_query:{}".format(predicted_sql_query))

            predicted_sql_query = self._add_from_clause(predicted_sql_query, world[i])
            # predicted_sql_query = ' '.join([token for token in predicted_sql_query_tokens])
            # print("predicted_sql_query:{}".format(predicted_sql_query))
            outputs['predicted_sql_query'].append(sqlparse.format(predicted_sql_query, reindent=False))

            if target_list is not None:
                targets = target_list[i].data
            target_available = target_list is not None and targets[0] > -1

            if target_available:
                sequence_in_targets = self._action_history_match(best_action_indices, targets)
                self._exact_match(sequence_in_targets)

                sql_evaluator_match = self._evaluate_func(original_gold_sql_query, predicted_sql_query, world[i].db_id)
                self._sql_evaluator_match(sql_evaluator_match)

                similarity = difflib.SequenceMatcher(None, best_action_indices, targets)
                self._action_similarity(similarity.ratio())

                difficulty = self._query_difficulty(targets, action_mapping, i)
                if difficulty:
                    self._acc_multi(sql_evaluator_match)
                else:
                    self._acc_single(sql_evaluator_match)

            beam_hit = False
            for pos, final_state in enumerate(best_final_states[i]):
                action_indices = final_state.action_history[0]
                action_strings = [action_mapping[(i, action_index)]
                                  for action_index in action_indices]
                candidate_sql_query = action_sequence_to_sql(action_strings, add_table_names=True)

                if target_available:
                    correct = self._evaluate_func(original_gold_sql_query, candidate_sql_query, world[i].db_id)
                    if correct:
                        beam_hit = True
                    self._beam_hit(beam_hit)
