import inspect
from badminton import landing_recovery as lr
source = inspect.getsource(lr._components).replace('def _components(', 'def _dark_components(')
source = source.replace('cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), max(1., sx)))', 'cv2.subtract(cv2.GaussianBlur(gray, (0, 0), max(1., sx)), gray)')
source = source.replace('(gray >= 100) & (saturation <= 160)', '(gray <= 220) & (saturation <= 190)')
source = source.replace("'native_bidirectional_temporal_component'", "'native_bidirectional_temporal_dark_component'")
ns=lr.__dict__.copy();exec(source,ns);dark=ns['_dark_components'];bright=lr._components
lr._components=lambda *args,**kwargs: bright(*args,**kwargs)+dark(*args,**kwargs)
