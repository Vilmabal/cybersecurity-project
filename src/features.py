#This contains all the functions that can be used to extract
#diffrent features out of a URL-link. 

from urllib.parse import urlparse, urljoin
import tldextract
from ipaddress import ip_address
import pandas as pd
from datetime import datetime, timezone
import requests
import socket
from bs4 import BeautifulSoup
from collections import Counter
from math import log2
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEYWORDS_FILE = PROJECT_ROOT/"data"/"phishing_keywords.xlsx"

keywords_df = pd.read_excel(KEYWORDS_FILE)


SUSPICIOUS_KEYWORDS = set(
    keywords_df["Evilginx Hostnames"]
    .dropna()
    .str.lower()
)

# Gives the length of the URL
def url_length(url: str) -> int:
    return len(url)

# Extracts the length of the domain out of URL
def domain_length(url: str) -> int:
    parsed = urlparse(url)
    domain = parsed.hostname
    return len(domain)

# Extracts number of dots out of URL
def number_of_dots(url: str) -> int:
    return url.count(".")

# Extracts number of subdomains out of URL
def number_of_subdomains(url: str) -> int:
    extracted = tldextract.extract(url)
    return len(extracted.subdomain.split(".")) if extracted.subdomain else 0


# Extracts number of hyphens out of URL
def number_of_hyphens(url: str) -> int:
    return url.count("-")


# Extracts number of @ symbols out of URL
def number_of_at_symbols(url: str) -> int:
    return url.count("@")


# Extracts number of digits out of URL
def number_of_digits(url: str) -> int:
    return sum(char.isdigit() for char in url)


# Checks if URL contains an IP address
def has_ip_address(url: str) -> bool:
    hostname = urlparse(url).hostname

    if hostname is None:
        return False

    try:
        ip_address(hostname)
        return True
    except ValueError:
        return False


# Checks if URL uses HTTPS
def is_https(url: str) -> bool:
    return urlparse(url).scheme == "https"


# Gives number of suspicious keywords in the domainname of the url
def suspicious_keyword_count(url: str) -> int:
    hostname = urlparse(url).hostname

    if hostname is None:
        return 0

    parts = hostname.lower().replace("-", ".").split(".")

    return sum(part in SUSPICIOUS_KEYWORDS for part in parts)


# Checks if domain uses Punycode
def has_punycode(url: str) -> bool:
    domain = urlparse(url).hostname

    if domain is None:
        return False

    return any(
        label.startswith("xn--")
        for label in domain.lower().split(".")
    )


# Extracts domain age in days using RDAP
def domain_age(url: str) -> int:
    domain = urlparse(url).hostname

    if domain is None:
        return -1

    try:
        response = requests.get(
            f"https://rdap.org/domain/{domain}",
            timeout=5
        )

        if response.status_code != 200:
            return -1

        data = response.json()

        for event in data.get("events", []):
            if event.get("eventAction") == "registration":
                creation_date = datetime.fromisoformat(
                    event["eventDate"].replace("Z", "+00:00")
                )

                return (datetime.now(timezone.utc) - creation_date).days

        return -1

    except (requests.RequestException, ValueError, KeyError):
        return -1

# Checks if domain has a DNS record
def dns_record_exists(url: str) -> bool:
    domain = urlparse(url).hostname

    if domain is None:
        return False

    try:
        socket.gethostbyname(domain)
        return True
    except socket.gaierror:
        return False


# Extracts number of external links from URL
def number_of_external_links(url: str) -> int:
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        base_domain = urlparse(url).hostname
        external_links = 0

        for link in soup.find_all("a", href=True):
            link_url = urljoin(url, link["href"])
            link_domain = urlparse(link_url).hostname

            if link_domain and link_domain != base_domain:
                external_links += 1

        return external_links

    except requests.RequestException:
        return 0


# Extracts number of password fields from URL
def number_of_password_fields(url: str) -> int:
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        password_fields = soup.find_all(
            "input",
            {"type": "password"}
        )

        return len(password_fields)

    except requests.RequestException:
        return 0


# Calculates entropy of letters in URL
def url_path_entropy(url: str) -> float:
    path = "".join(
        char.lower()
        for char in urlparse(url).path
        if char.isalnum()
    )

    if not path:
        return 0.0

    frequencies = Counter(path)
    length = len(path)

    return -sum(
        (count / length) * log2(count / length)
        for count in frequencies.values()
    )