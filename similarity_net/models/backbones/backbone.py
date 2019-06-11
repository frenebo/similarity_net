
class Backbone:
    # For when backbone has custom objects, like special layers
    custom_objects = {}

    def __init__(self):
        pass

    def call_on_inputs(self, inputs):
        """ Should return backbone output layers given the inputs
        """
        raise NotImplementedError("Unimplemented layer")