def main():
    #TODO: FIX THE ARGUMENTS TO FIT THE DATASET (URMP and R2D2)
    parser = argparse.ArgumentParser(description='Train log-mel-to-mask network')
    parser.add_argument('--dataset_train_path', type=str, help='Folder containing Train/val Dataset audio',
                        default='dataset/starnet/starnet_reduced')
    parser.add_argument('--desired_instrument', type=str, help='Desired Output Timbre',
                        default='strings')
    parser.add_argument('--conditioning_instrument', type=str, help='Desired Conditioning Timbre',
                        default='clarinet')
    parser.add_argument('--GPU', type=str, help='Select GPU number',
                        default='0')
    parser.add_argument('--train', type=str, help='Select GPU number',
                        default='True')


    dict_instruments = {"clarinet":"1","vibraphone":"2","strings":"4","piano":"5",'clarinet_vibraphone':"0",'strings_piano':"3"}
    
    args = parser.parse_args()

    desired_instrument = args.desired_instrument
    conditioning_instrument = args.conditioning_instrument
    dataset_train_path = args.dataset_train_path
    train = args.train
    print('Timbre transfering from '+conditioning_instrument+' to'+desired_instrument)

    # Handle Paths
    instruments_name = [dict_instruments[desired_instrument],dict_instruments[conditioning_instrument]]

    checkpoint_path = "checkpoints/ATT_STARNET_NORM_diffusion_model_timbre_transfer_"+conditioning_instrument+'_to_'+desired_instrument+'_'+ datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/ATT_STARNET_"+conditioning_instrument+'_to_'+desired_instrument+datetime.datetime.now().strftime(
    #     "%Y%m%d-%H%M%S"))

    log_dir = 'logs/'
    logdir = log_dir + 'ATT_STARNET_NORM_diffusion_timbre_transfer_' + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + '5000__'+conditioning_instrument+'_to_'+desired_instrument


    # TODO: REPLACE WITH THE CUSTOM PATH AND PARSING TO THE RIGHT PLACE IN INPUT
    # Each instrument is the same since track names are duplicatedk

    # tracks_full = os.listdir(dataset_train_path)
    # cond_tracks = [track for track in tracks_full if track.split('.')[-2]==instruments_name[1]]
    # trgt_tracks = [track for track in tracks_full if track.split('.')[-2]==instruments_name[0]]
    # cond_tracks.sort()
    # trgt_tracks.sort()
    # track_paths_trans = [[os.path.join(dataset_train_path,trgt_tracks[i]),os.path.join(dataset_train_path,cond_tracks[i])] for i in range(len(cond_tracks))]


    # val_perc = 0.2
    # n_tracks_train = len(track_paths_trans) - int(np.floor(val_perc * len(track_paths_trans)))
    # rng = np.random.default_rng(12345)
    # idxs = rng.choice(len(track_paths_trans), len(track_paths_trans), False)
    # idxs_train = idxs[:n_tracks_train]
    # idxs_val = idxs[n_tracks_train:]

    # train_tracks_paths = np.array(track_paths_trans)[idxs_train].tolist()
    # val_tracks_paths = np.array(track_paths_trans)[idxs_val].tolist()

    # train_dataset = prepare_dataset(train_tracks_paths)
    # val_dataset = prepare_dataset(val_tracks_paths, training=False)


    # create and compile the model
    first = True
    for a in val_dataset.take(2):
        if first:
            val_data = a
            first = False
        else:
            val_data = tf.concat([val_data, a],axis=0)
    val_data = a[:18]
    print(val_data.shape)
    model = network_lib.DiffusionModel(params.mel_spec_size, params.widths, params.block_depth, val_data, params.has_attention, logdir=logdir,batch_size=params.batch_size,)
    model.network.summary()

    if train:
        model.compile(
            optimizer=keras.optimizers.experimental.AdamW(
                learning_rate=params.learning_rate, weight_decay=params.weight_decay
            ),
            loss=keras.losses.mean_absolute_error,
        )

        # save the best model based on the validation KID metric

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor="val_n_loss",
            mode="min",
            save_best_only=True,
        )

        # calculate mean and variance of training dataset for normalization
        model.fit(
            train_dataset,
            epochs=5000,
            validation_data=val_dataset,
            callbacks=[
                keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
                checkpoint_callback,
                tensorboard_callback,
            ],
        )

if __name__=='__main__':
    main()
