def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.chop = True

    if args.template.find('RCAN_wo_CA') >= 0:
        args.model = 'RCAN_wo_CA'
        args.chop = True
