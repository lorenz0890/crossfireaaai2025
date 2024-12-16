import warnings

from torch.nn import Module

DEFAULT_WRAPPED_OBJECT_NAME = "_wrapped_object"


class BaseWrapper(Module):
    """
    Basic class to cover or enclose any object and store it in private attribute `_wrapped_object`. All the attributes
    of the wrapped object can be accessed from the BaseWrapper instance, and if there's any overlap (wrapped class
    and object instance shared the same name for a variable or function) between the wrapped class and instance object,
    then the priority will be the variable or the function from the wrapper.

    source: https://code.activestate.com/recipes/577555-object-wrapper-class/
    """

    def __init__(self, obj):
        """
        :param obj: **any** instance object to wrap it
        """
        super(BaseWrapper, self).__init__()
        self.__setattr__(DEFAULT_WRAPPED_OBJECT_NAME, obj)

    @staticmethod
    def unwrap(obj):
        """
        Remove all the quantization during training functions, and allow deployment.
        """
        warnings.warn("`unwrap` function is not implemented in the current scope.")
        return obj

    @staticmethod
    def deconstruct_wrap(obj):
        """
        Unwrapping function to remove the all the wrapped operation and have a native instance of the provided object
        without changing anything of the instance attributes

        :return: the native instance object without any additional attributes from the wrapper
        """
        # if the object has `_wrapped_object`, store the `wrapped_object` in the object
        if hasattr(obj, DEFAULT_WRAPPED_OBJECT_NAME):
            obj = getattr(obj, DEFAULT_WRAPPED_OBJECT_NAME)
        return obj


class TorchModuleWrapper(BaseWrapper):
    """
    Wrap the object instance that is inherent from `torch.nn.Module` by forwarding the `__call__` function to the
    implementation of the call inside the instance object, and rename the class name to `TorchModuleWrapper`.
    """

    def __init__(self, torch_object: Module):
        super(TorchModuleWrapper, self).__init__(torch_object)

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._wrapped_object(*args, **kwargs)

    def __repr__(self):
        return "".join([self.__class__.__name__, self._wrapped_object.__repr__()])

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = self
            memo[id(self)] = result
            return result
