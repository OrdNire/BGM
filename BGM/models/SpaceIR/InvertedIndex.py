
class InvertedIndex:
    def __init__(self):
        self.index = {}  # 倒排索引字典
        self.content = {}  # 图像的label信息
        self.content2image = {}

    def add_content(self, img_id, content):
        """
        :param img_id: 图像ID
        :param content: 图像的label信息, type = list
        """
        self.content[img_id] = content
        # label_key = ",".join(map(str, content))
        # if label_key not in self.content2image:
        #     self.content2image[label_key] = [img_id]
        # else:
        #     self.content2image[label_key].append(img_id)
        for label in content:
            if label not in self.index.keys():
                self.index[label] = set()
            self.index[label].add(img_id)

    def add_contentV2(self, img_id, content):
        """
        :param img_id: 图像ID
        :param content: 图像的label信息, type = list
        """
        self.content[img_id] = content
        # label_key = ",".join(map(str, content))
        # if label_key not in self.content2image:
        #     self.content2image[label_key] = [img_id]
        # else:
        #     self.content2image[label_key].append(img_id)
        for label, score in content:
            if label not in self.index.keys():
                self.index[label] = set()
            self.index[label].add((img_id, score))

    def search(self, query):
        query_list = list(query)
        result = set()
        next_result = set(self.content.keys())
        for q in query_list:
            if q in self.index.keys():
                next_result &= self.index[q]
                if len(next_result) > 0:
                    result = next_result.copy()
        return result

    def search_many(self, many_query):
        res_list = []
        for query in many_query:
            res = self.search(query)
            res_list.extend(res)
        return res_list

    # 交集
    def search_union(self, query_list):
        res = set()
        for q in query_list:
            if q in self.index:
                res |= self.index[q]
        return res

    def search_content(self, img_id):
        return self.content[img_id]

    def display_index(self):
        """
        打印倒排索引
        """
        print("ividx")
        for label, img_ids in self.index.items():
            print(f"'{label}': {img_ids}")
        print("idx")
        for label_key, img_ids in self.content2image.items():
            print(f"{label_key}:{img_ids}")

if __name__ == '__main__':
    ividx = InvertedIndex()
    ividx.add_content("1,5", [1, 2])
    ividx.add_content("2,4", [3, 4])
    ividx.add_content("3,7", [1, 3])
    res = ividx.search([3])
    print(res)