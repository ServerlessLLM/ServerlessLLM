import os
import shutil


def is_folder_accessible(folder_path):
    """
    Check if the folder exists and if it's accessible (read, write, execute).

    :param folder_path: The path to the folder.
    :return: True if the folder is accessible, False otherwise.
    """
    # Check if the path exists and is a directory
    if not os.path.isdir(folder_path):
        return False

    # Check for read, write, and execute permissions
    can_read = os.access(folder_path, os.R_OK)
    can_write = os.access(folder_path, os.W_OK)
    can_execute = os.access(folder_path, os.X_OK)

    # The folder is accessible if all permissions are granted
    return can_read and can_write and can_execute


def main():
    home_dir = os.path.expanduser("~")
    model_path = os.path.join(home_dir, "models")
    if is_folder_accessible(model_path):
        print("The '~/models' folder exists and is accessible.")
    else:
        print("Access issue detected with '~/models' folder.")
        if os.path.exists(model_path):
            print("Attempting to delete the inaccessible '~/models' folder...")
            try:
                shutil.rmtree(model_path)
                print("Inaccessible folder deleted successfully.")
            except PermissionError:
                print("Permission denied: Cannot delete the '~/models' folder.")
                return
            except Exception as e:
                print(f"An error occurred while deleting the folder: {e}")
                return
        else:
            print(
                "The '~/models' folder does not exist. No deletion necessary."
            )

        # Attempt to create the folder after deletion
        try:
            os.makedirs(model_path)
            print("The '~/models' folder has been successfully created.")
        except PermissionError:
            print("Permission denied: Cannot create the '~/models' folder.")
        except Exception as e:
            print(f"An error occurred while creating the folder: {e}")


if __name__ == "__main__":
    main()
