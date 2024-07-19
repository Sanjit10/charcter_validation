class BaseProcessor:
    def process(self, data):
        raise NotImplementedError("Process method not implemented.")

class AProcessor(BaseProcessor):
    def process(self, data):
        # Processing logic for detection class 'A'
        return f"Processed {data} with AProcessor"

class BProcessor(BaseProcessor):
    def process(self, data):
        # Processing logic for detection class 'B'
        return f"Processed {data} with BProcessor"

class ProcessorFactory:
    _processors = {
        'A': AProcessor(),
        'B': BProcessor(),
        # Add more mappings as needed
    }

    @staticmethod
    def get_processor(detection_class):
        processor = ProcessorFactory._processors.get(detection_class)
        if not processor:
            raise ValueError(f"No processor found for detection class {detection_class}")
        return processor
