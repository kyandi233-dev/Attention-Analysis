from nir_supervisor import command_key, subject_from_command


def test_command_key_ignores_windows_python_executable():
    a = r'"D:\env\Scripts\python.exe" run_pipeline.py --video J:\Data\sub-082_\nir\sub-082_nir.avi'
    b = r'G:\Python\python.exe run_pipeline.py --video J:\Data\sub-082_\nir\sub-082_nir.avi'
    assert command_key(a) == command_key(b)


def test_subject_from_pipeline_command():
    command = r'python run_pipeline.py --video J:\Data\sub-082_\nir\sub-082_nir.avi'
    assert subject_from_command(command) == "sub-082"


def test_subject_from_non_subject_command_is_none():
    assert subject_from_command("python run_formal_batch.py --backend pytorch-cuda") is None
