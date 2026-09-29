#This contains all the functions that can be used to extract
#diffrent features out of a URL-link. 

import urllib
from urllib.parse import urlparse
import tldextract
from ipaddress import ip_address
import pandas as pd
from datetime import datetime, timezone
import requests


# possible features:
# URL length - ok
# Domain length - ok
# Number of dots - ok
# Number of subdomains - ok
# Number of hyphens - ok 
# Number of @ symbols - ok 
# Number of digits in URL - ok
# IP address in URL (binary) 
# HTTPS scheme (binary) 
# Suspicious keyword count 
# Punycode in domain (binary) (xn--) 
# Domain age (days) -> WHOIS 
# DNS record exists (binary) -> dns reslover 
# Number of external links 
# Number of password fields
# page-rank
# entropy of letters in link


keywords_df = pd.read_excel("data/phishing_keywords.xlsx")

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
    domain = parsed.netloc
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

    return "xn--" in domain.lower()


# Extracts domain age in days using WHOIS
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
def dns_record_exists(url: str) -> int:
    ...


# Extracts number of external links from URL
def number_of_external_links(url: str) -> int:
    ...


# Extracts number of password fields from URL
def number_of_password_fields(url: str) -> int:
    ...


# Extracts PageRank of URL
def page_rank(url: str) -> float:
    ...


# Calculates entropy of letters in URL
def url_letter_entropy(url: str) -> float:
    ...