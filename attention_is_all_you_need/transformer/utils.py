


def total_parameters(model):
    total_params = sum(
	param.numel() for param in model.parameters()
    )   
    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total Parameters: {total_params}, Trainable Parameters: {trainable_params}")

