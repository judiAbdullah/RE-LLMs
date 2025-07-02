class SeqCleaner:
    def clean_element(self, element, head=None, scoped=[]):
        if head is None:
            head = element
        seq = []
        for elt in element:
            try:
                if elt["type"] == "methodInvocation":
                    seq.append(elt)
                elif elt["type"] == "newInstance":
                    if "name" in elt:
                        if self.is_used(elt["name"], head):
                            seq.append(elt)
                            scoped.append(elt["name"])

                elif elt["type"] == "scopedVariable":
                    if self.is_used(elt["name"], head) and elt["name"] not in scoped:
                        seq.append(elt)
                        scoped.append(elt["name"])

                elif elt["type"] == "controlFlow":
                    seq.append(elt)

                elif elt["type"] == "blocks":
                    for b in elt["blocks"]:
                        b["contents"] = self.clean_element(b["contents"], head=head, scoped=scoped)
                    seq.append(elt)
                else:
                    raise Exception("unknown type", elt["type"])

            except Exception as e:
                print("draw exception:", e)
                continue
        return seq

    def is_used(self, name, elements):
        for elt in elements:
            if elt["type"] == "methodInvocation":
                if name in elt["to"]:
                    return True

            elif elt["type"] == "blocks":
                for block in elt["blocks"]:
                    if self.is_used(name, block["contents"]):
                        return True

        return False