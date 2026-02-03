import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os
import time

def download_papers():
    # 1. Setup Directories
    # Get the absolute path of the script and resolve the target directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, './papers')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # 2. Construct the API Query
    # Query: Categories (Computation & Language, Machine Learning, AI) AND keywords
    # We use 'cat:cs.CL' (Comp Ling/LLMs), 'cs.LG' (Learning), 'cs.AI' (AI)
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = 'cat:cs.CL OR cat:cs.LG OR cat:cs.AI'
    
    params = {
        'search_query': search_query,
        'start': 0,
        'max_results': 100,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    query_string = urllib.parse.urlencode(params)
    api_url = base_url + query_string
    
    print(f"Fetching metadata from: {api_url}")

    # 3. Fetch Metadata
    try:
        with urllib.request.urlopen(api_url) as response:
            xml_data = response.read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching API data: {e}")
        return

    # 4. Parse XML and Download PDFs
    # XML Namespaces are required for ElementTree parsing of Atom feeds
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(xml_data)
    
    entries = root.findall('atom:entry', ns)
    print(f"Found {len(entries)} papers. Starting download...")
    
    downloaded_count = 0

    for i, entry in enumerate(entries):
        try:
            # Get ID to use as filename (e.g., http://arxiv.org/abs/2101.12345 -> 2101.12345)
            id_url = entry.find('atom:id', ns).text
            paper_id = id_url.split('/')[-1]
            
            # Find the PDF link
            pdf_link = None
            for link in entry.findall('atom:link', ns):
                if link.attrib.get('title') == 'pdf':
                    pdf_link = link.attrib['href']
                    break
            
            # Fallback if specific pdf link tag isn't found, construct it
            if not pdf_link:
                pdf_link = f"http://arxiv.org/pdf/{paper_id}.pdf"

            # Create clean filename
            filename = f"{paper_id}.pdf"
            filepath = os.path.join(save_dir, filename)

            # Skip if already exists
            if os.path.exists(filepath):
                print(f"[{i+1}/{len(entries)}] Skipping {filename} (already exists)")
                continue

            # Download
            print(f"[{i+1}/{len(entries)}] Downloading {filename}...")
            # Custom header often helps avoid 403 errors on pdf download
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(pdf_link, filepath)
            downloaded_count += 1
            
            # Be polite to the server
            time.sleep(3) 

        except Exception as e:
            print(f"Failed to download paper {i+1}: {e}")

    # 5. Final Report
    print("-" * 30)
    print(f"Process Complete.")
    print(f"Total papers downloaded: {downloaded_count}")
    print(f"Saved to: {save_dir}")

if __name__ == "__main__":
    download_papers()