from typing import Dict, Any, List, Tuple

import overrides
import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, TimeDistributed, Attention, Embedding, \
    FeedForward
from allennlp.nn import util, Activation
from allennlp.state_machines.states import RnnStatelet, GrammarStatelet
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.state_machines.transition_functions import LinkingTransitionFunction

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
                 input_attention: Attention,
                 past_attention: Attention,
                 action_embedding_dim: int,
                 max_decoding_steps: int,
                 nhop: int,
                 vocab: Vocabulary,
                 training_beam_size: int = None,
                 add_action_bias: bool = True,
                 decoder_self_attend: bool = True,
                 decoder_num_layers: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__(vocab)

        self.question_embedder = question_embedder
        self._input_mm_embedder = input_memory_embedder
        self._output_mm_embedder = output_memory_embedder
        self._question_encoder = question_encoder
        self._input_mm_encoder = TimeDistributed(input_memory_encoder)
        self._output_mm_encoder = TimeDistributed(output_memory_encoder)

        self._self_attend = decoder_self_attend
        self._max_decoding_steps = max_decoding_steps
        self._add_action_bias = add_action_bias
        num_actions = vocab.get_vocab_size(self._rule_namespace)
        if self._add_action_bias:
            input_action_dim = action_embedding_dim + 1
        else:
            input_action_dim = action_embedding_dim
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=input_action_dim)
        self._output_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)

        # TODO
        self._num_entity_types = -1
        self._entity_type_encoder_embedding = Embedding(self._num_entity_types, question_encoder.get_output_dim())

        self._decoder_num_layers = decoder_num_layers
        self._action_embedding_dim = action_embedding_dim

        # TODO
        self._ent2ent_ff = FeedForward(action_embedding_dim, 1, action_embedding_dim, Activation.by_name('relu')())

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_utterance = torch.nn.Parameter(
            torch.FloatTensor(encoder_output_dim=question_encoder.get_output_dim()))

        if self._self_attend:
            self._transition_function = AttendPastSchemaItemsTransitionFunction(
                encoder_output_dim=question_encoder.get_output_dim(),
                action_embedding_dim=action_embedding_dim,
                input_attention=input_attention,
                past_attention=past_attention,
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
        self._decoder_trainer = MaximumMarginalLikelihood(training_beam_size)

    @overrides
    def forward(self,
                worlds: List[SpiderWorld],
                valid_actions: List[List[ProductionRule]],
                utterance: Dict[str, torch.LongTensor],
                schema: Dict[str, torch.LongTensor],
                action_sequence: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        device = utterance['tokens'].device
        initial_state = self._get_initial_state(utterance, worlds, schema, valid_actions)

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

            # TODO
            # self._compute_validation_outputs(valid_actions,
            #                                  best_final_states,
            #                                  world,
            #                                  action_sequence,
            #                                  outputs)
        return outputs

    def _get_initial_state(self,
                           utterance: Dict[str, torch.LongTensor],
                           worlds: List[SpiderWorld],
                           schema: Dict[str, torch.LongTensor],
                           valid_actions: List[List[ProductionRule]]) -> GrammarBasedState:

        utterance_mask = util.get_text_field_mask(utterance).float()
        embedded_utterance = self.question_embedder(utterance)
        batch_size, _, _ = embedded_utterance.size()
        encoder_outputs = self._dropout(self.encoder(embedded_utterance, utterance_mask))

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
        K = self._input_mm_encoder(input_mm_schema, schema_mask)
        V = self._output_mm_encoder(output_mm_schema, schema_mask)

        encoder_output_dim = self._encoder.get_output_dim()

        # Encodes utterance in the context of the schema, which is stored in external memory
        encoder_outputs_with_context, attn_weights = self._mm_attn(encoder_outputs, K, V)
        final_encoder_output = util.get_final_encoder_states(encoder_outputs_with_context,
                                                             utterance_mask,
                                                             self._encoder.is_bidirectional())
        # TODO
        if self._self_attend:
            entities_ff = self._ent2ent_ff(V)
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

        initial_sql_state = [SqlState(valid_actions[i], worlds[i].db_context.schema, self.parse_sql_on_decoding) for i
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
                global_output_embeddings = self._output_action_embedder(global_action_tensor)
                translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                           global_output_embeddings,
                                                           list(global_action_ids))
            if linked_actions:
                linked_rules, linked_action_ids = zip(*linked_actions)
                entities = [rule.split(' -> ')[1].strip('[]\"') for rule in linked_rules]

                entity_ids = [entity_map[entity] for entity in entities]

                entity_linking_scores = attn_weights[entity_ids]

                if linked_actions_linking_scores is not None:
                    entity_action_linking_scores = linked_actions_linking_scores[entity_ids]

                # if not self._decoder_use_graph_entities:
                entity_type_tensor = entity_types[entity_ids]
                entity_type_embeddings = (self._entity_type_decoder_embedding(entity_type_tensor)
                                          .to(entity_types.device)
                                          .float())
                # else:
                #     entity_type_embeddings = entity_graph_encoding.index_select(
                #         dim=0,
                #         index=torch.tensor(entity_ids, device=entity_graph_encoding.device)
                #     )

                if self._self_attend:
                    translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                               entity_type_embeddings,
                                                               list(linked_action_ids),
                                                               entity_action_linking_scores)
                else:
                    translated_valid_actions[key]['linked'] = (entity_linking_scores,
                                                               entity_type_embeddings,
                                                               list(linked_action_ids))

        return GrammarStatelet(['statement'],
                               translated_valid_actions,
                               self.is_nonterminal)
