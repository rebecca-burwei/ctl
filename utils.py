import xml.etree.ElementTree as ET
import pandas as pd

def xml_to_df(path_to_xml):
    """Loads an XML file from Stack Exchange as a pandas dataframe."""
    tree = ET.parse(path_to_xml)
    root = tree.getroot()
    children = [child.attrib for child in root]
    return pd.DataFrame(children)
