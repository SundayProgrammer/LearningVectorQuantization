class InvalidParameterException(Exception):
    msg = None
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def message(self):
        return str(self)
        
    def __str__(self):
        msg = self.msg.format(**self.kwargs)
        return msg 

class EmptyListException(Exception):
    pass

