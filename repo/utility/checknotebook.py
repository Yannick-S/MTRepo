def in_ipynb():
    try:
        cfg = get_ipython().config 
        if 'jupyter' in cfg['IPKernelApp']['connection_file']:
            return True
        else:
            return False
    except NameError:
        return False
    
if __name__ == "__main__":
    print(in_ipynb())