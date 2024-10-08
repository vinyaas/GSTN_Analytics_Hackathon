import hashlib
import argparse

def calculate_checksum(file_path, hash_algorithm='sha256'):
    try:
        hash_func = hashlib.new(hash_algorithm)
        with open(file_path, 'rb') as file:
            while chunk := file.read(8192):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except ValueError:
        return f"Error: Invalid hash algorithm '{hash_algorithm}'."
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Verify the checksum of a file.")
    parser.add_argument("file_path", type=str, help="Path to the file to be checksummed.")
    parser.add_argument("-a", "--algorithm", type=str, choices=['md5', 'sha1', 'sha256', 'sha512'], default='sha256', help="Hash algorithm to use (default: sha256).")

    args = parser.parse_args()
    file_path = args.file_path
    hash_algorithm = args.algorithm

    checksum = calculate_checksum(file_path, hash_algorithm)
    print(f"{hash_algorithm.upper()} Checksum: {checksum}")

if __name__ == "__main__":
    main()
