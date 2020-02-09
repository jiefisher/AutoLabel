class Params:
    ACT_NUM_EPOCHS = 50
    CRITIC_NUM_EPOCHS=50
    ALPHA = 5e-3        # learning rate
    TRAIN_BATCH_SIZE = 3     # how many episodes we want to pack into an epoch
    TEST_BATCH_SIZE = 3
    HIDDEN_DIM = 300    # number of hidden nodes we have in our dnn
    EMBED_DIM = 300
    BETA = 0.1          # the entropy bonus multiplier