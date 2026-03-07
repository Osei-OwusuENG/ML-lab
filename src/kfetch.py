import kaggle
import argparse
import pandas as pd
import sys


# Format Function
def format_size(ds):
    #Get size
    s = getattr(ds, 'size', None)
    
    if s: return s

    else:
        #Get byte and convert to MB
        byte_val = getattr(ds, 'total_bytes', 0)
        return f"{byte_val/(1024*1024):.3f} MB"


# kaggle data search function
def search_name(search):

    result = kaggle.api.dataset_list(search=search)

    data = {
        "ref": [getattr(ds, 'ref', 'N/A') for ds in result],
        "voteCount": [getattr(ds, 'vote_count', 0) for ds in result],
        "usabilityRating": [round(getattr(ds, 'usability_rating', 0), 3) for ds in result],
        "downloadCount": [getattr(ds, 'download_count', 0) for ds in result],
        "size": [format_size(ds) for ds in result],
        "lastUpdated": [str(getattr(ds, 'last_updated', 'N/A'))[:10] for ds in result]
    }

    return pd.DataFrame(data).set_index('ref')

# kaggle download data function
def download_name(search, path, unzip_f: bool):
    print("connection to kaggle to download data")

    try:
        kaggle.api.dataset_download_files(search, path=path, unzip=unzip_f)
        print(f"File successfully downloaded to path: {path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':

    # authenticate kaggle
    kaggle.api.authenticate()

    parser = argparse.ArgumentParser(description="Kaggle Tool")

    # add arguments to the command line
    parser.add_argument("--search", help="Search term")
    parser.add_argument("--path", default='./data', help="Download path")
    parser.add_argument("--rows", type=int, default=5, help="Rows to return")
    parser.add_argument("-H", "--head", action='store_true', help="Return specified rows")
    parser.add_argument("-r", "--rating", action='store_true', help="Sort by usabilityRating")
    parser.add_argument("-v", "--verbose", action="store_true", help="Switch to Download mode")
    parser.add_argument("-u", "--unzip_f", action="store_false", help="Unzip file")
    
    args = parser.parse_args()
  
        
    # check search term
    if not args.search:
        print('provide a search term')
        sys.exit()
    
    
    if not args.verbose:

        # SEARCH MODE
        df = search_name(args.search)

        # DETERMINE SORT COLUMN
        sort_col = 'usabilityRating' if args.rating else 'voteCount'

        df = df.sort_values(by=sort_col, ascending=False)

        # return rows if provided
        if args.head:
            df = df.head(args.rows)
        
        pd.set_option('display.max_colwidth', None)
        print(f'\n{df}')
    
    else:
        #Download Mode
        df = download_name(args.search, args.path, args.unzip_f)
        
    
    