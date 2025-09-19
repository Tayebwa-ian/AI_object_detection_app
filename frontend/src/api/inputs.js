const useDummy = true;

export const uploadImage = async (file) => {
  if (useDummy) {
    return new Promise((res) =>
      setTimeout(() => res({ id: "dummy-input-id", name: file.name, url: URL.createObjectURL(file) }), 500)
    );
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/v1/inputs", { method: "POST", body: formData });
  return response.json();
};
