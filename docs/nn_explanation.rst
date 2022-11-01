These parameters are
            disseminated into four categories.

            * Neural network parameters

                - model_choice : str, model choice, either "1DCNN" or
                "LSTM_multihead"
                - params : list[str], column names in train & test
                dataframes, used in neural network. Each value in this column
                must be a list.
                - dropout_pct : float, rate at which to set input units
                to zero.
                - verbose : int, control the specificity of the prints.

            * Training parameters

                - train_size : float, split of training data used for
                training
                - **shuffle_split** (*bool*), shuffle data during test-train
                split
                - **balance_tactic** (*str*), mode balancing tactic, either
                "truncate" or "gravitate". Truncate will utilize the exact
                same number of samples for each category. Gravitate will sway
                the original number of samples towards the same number.
                Default= truncate.
                - **n_split** (*int*), number of splits in the stratified KFold
                cross validation.
                - **batch_size** (*int*), number of samples per gradient update.
                - **max_epochs** (*int*), maximum number of passes through the
                training process.

            * LSTM parameters

                - **use_attention_lstm** (*bool*), if True,
                use attention in LSTM network
                - **units** (*int*), number of neurons in initial NN layer

                * 1DCNN parameters

                - **nfilters** (*int*), number of filters in the convolution.
                - **kernel_size** (*int*), length of the convolution window.