def get_config_value(config, key, default_value):
    """Retrieve a configuration value while providing a default fallback."""
    return config.get(key, default_value)  # Returns value if present, otherwise the provided default


def validate_class_config(condition_config):
    """Validate that the class conditioning configuration is complete."""
    assert "class_condition_config" in condition_config, "Class conditioning desired but class condition config missing."  # Ensures nested config exists
    assert "num_classes" in condition_config["class_condition_config"], "num_classes missing in class condition config."  # Verifies class count is provided


def validate_text_config(condition_config):
    """Validate that the text conditioning configuration is complete."""
    assert "text_condition_config" in condition_config, "Text conditioning desired but text condition config missing."  # Ensures nested text config exists
    assert "text_embed_dim" in condition_config["text_condition_config"], "text_embed_dim missing in text condition config."  # Verifies embedding dimension is specified


def validate_class_conditional_input(cond_input, x, num_classes):
    """Ensure runtime class conditioning tensors are well-formed."""
    assert cond_input is not None, "cond_input must be provided when class conditioning is enabled."  # Guard against missing conditioning dictionary
    assert "class" in cond_input, "cond_input must contain a 'class' entry for class conditioning."  # Require class logits or one-hot vectors
    class_tensor = cond_input["class"]  # Extracts the tensor carrying class conditioning information
    assert class_tensor.shape == (x.shape[0], num_classes), "Class conditioning tensor must have shape (batch_size, num_classes)."  # Enforces expected tensor shape
