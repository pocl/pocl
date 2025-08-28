
typedef std::map<const char*, std::string> ReplaceMapT;

void replaceAllStringsInMap(std::string &Buffer, ReplaceMapT RepMap);

const char* dtype2precision(cl_tensor_datatype_exp dtype);

const char* dtype2elemtype(cl_tensor_datatype_exp dtype);

const char *layout2str (const cl_tensor_desc_exp &tensor);

bool instantiateTemplateMATMUL(const void* KernelAttrs,
                                    std::string &ModelXMLInstance,
                                    std::string &BuildFlagsInstance);

bool instantiateTemplateGEMM(const void* KernelAttrs,
                             std::string &ModelXMLInstance,
                             std::string &BuildFlagsInstance);
