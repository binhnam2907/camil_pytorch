
import sys
import os
import sys 

PRJ_DIR = os.environ.get("PROJECT_DIR")
sys.path.append(os.path.join(PRJ_DIR))
#sys.path.append(os.path.join(PRJ_DIR, "nystromformer"))

from nystromformer.nystrom_attention import NystromAttention  
from nystromformer.nystrom_former import Nystromformer



if __name__ == "__main__":
    print(">> dome import ")
