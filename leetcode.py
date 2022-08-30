# # def findStar(stars):
# #     n = len(stars)
# #     visited = []
# #     res = 0
# #     for node1 in range(n - 2):
# #         for node2 in range(node1 + 1, n - 1):
# #             x = stars[node1][0] - stars[node2][0]
# #             y = stars[node1][1] - stars[node2][1]
# #             cnt = 2
# #             if (x, y) not in visited:
# #                 visited.append((x, y))
# #                 for node3 in range(node2 + 1, n):
# #                     x1 = stars[node1][0] - stars[node3][0]
# #                     y1 = stars[node1][1] - stars[node3][1]
# #                     if x * y1 == x1 * y:
# #                         cnt += 1
# #             res = max(res, cnt)
# #     return res
# #
# #
# # n = int(input())
# # stars = []
# # for i in range(n):
# #     data = list(map(int, input().strip().split()))
# #     stars.append(data)
# # print(findStar(stars))
# # #
# # # stars = [[1, 1], [2, 2], [3, 5]]
# # # # a = stars.pop()
# # # # stars = set(stars)
# # # # b = stars.remove((2, 3))
# # # # print(a)
# # # # print(b)
# # # # print(stars)
# # # print(findStar(stars))

# # def candy(ratings ):
# #     n=len(ratings)
# #     candys=[1]*n
# #     for i in range(n-1):
# #         if ratings[i+1]>ratings[i]:
# #             candys[i+1]=candys[i]+1
# #     for j in range(n-1,0,-1):
# #         if ratings[j-1]>ratings[j]:
# #             candys[j-1]=max(candys[j]+1,candys[j-1])
# #     return sum(candys)
# #
# # ratings=[1,2,3,4,0]
# # print(candy(ratings))

# import sys

# # def mapMoving(data, ind, do):
# #     row, col = len(data), len(data[0])
# #     neighbor = {'L': [0, -1], 'R': [0, 1], 'D': [-1, 0], 'U': [1, 0]}
# #     visited = []
# #     for i in range(row):
# #         for j in range(col):
# #             if data[i][j] == ind:
# #                 next_i = neighbor[do][0] + i
# #                 next_j = neighbor[do][1] + j
# #                 if 0 <= next_i < row and 0 <= next_j < col and (
# #                         data[next_i][next_j] == 0 or data[next_i][next_j] == ind):
# #                     visited.append((next_i, next_j))
# #                     data[i][j] = 0
# #                 else:
# #                     return []
# #     for i, j in visited:
# #         data[i][j] = ind
# #     return data
# #
# #
# # #
# # if __name__ == "__main__":
# #     # 读取第一行的n
# #     # n, k = list(map(int, sys.stdin.readline().strip().split()))
# #     n, k = 2, 4
# #     data = []
# #     # for i in range(n):
# #     #     # 读取每一行
# #     #     line = sys.stdin.readline().strip()
# #     #     # 把每一行的数字分隔后转化成int列表
# #     #     values = list(map(int, line.split()))
# #     #     data.append(values)
# #     #     print(data)
# #     data = [[1, 0, 0, 0, 0, 0, 0, 0],
# #             [1, 0, 0, 4, 4, 0, 0, 0],
# #             [1, 0, 0, 4, 0, 0, 0, 0],
# #             [0, 0, 0, 4, 0, 0, 0, 0],
# #             [0, 0, 4, 4, 0, 2, 2, 0],
# #             [0, 0, 0, 4, 0, 0, 2, 0],
# #             [0, 0, 0, 4, 4, 0, 0, 3],
# #             [0, 0, 0, 0, 0, 0, 0, 0]]
# #     for j in range(k):
# #         line = sys.stdin.readline().strip().split()
# #         ind = int(line[0])
# #         do = line[1]
# #         # print(do)
# #         res = mapMoving(data, ind, do)
# #         if res:
# #             data = res
# #         print(res)
# #
# #     for i in range(len(data)):
# #         for j in range(len(data[0])):
# #             print(data[i][j], end='')
# #         print('')
# # neighbor = {'L': [0, -1], 'R': [0, 1], 'D': [-1, 0], 'U': [1, 0]}
# # print(neighbor['L'])

# import sys

# def mapMoving(data, ind, do):
#     row, col = len(data), len(data[0])
#     neighbor = {'L': [0, -1], 'R': [0, 1], 'D': [1, 0], 'U': [-1, 0]}
#     visited = []
#     def findInd():
#         for i in range(row):
#             for j in range(col):
#                 if data[i][j] == ind:
#                     return i,j
#     i,j=findInd()
#     stack = [(i, j)]
#     print(stack)
#     next = [(-1, 0), (1, 0), (0, 1), (0, -1)]
#     while stack:
#         i, j = stack.pop()
#         visited.append((i, j))
#         for delt_x, delt_y in next:
#             next_i = delt_x + i
#             next_j = delt_y + j
#             if 0 <= next_i < row and 0 <= next_j < col and (next_i, next_j) not in visited and data[next_i][
#                 next_j] == ind:
#                 stack.append((next_i, next_j))
#                 print(visited)

#     if do == 'L':
#         visited.sort(key=lambda x: x[1])
#     elif do == 'R':
#         visited.sort(key=lambda x: -x[1])
#     elif do == 'D':
#         visited.sort(key=lambda x: -x[0])
#     else:
#         visited.sort(key=lambda x: x[0])
#     for i, j in visited:
#         # print(visited)
#         next_i = neighbor[do][0] + i
#         next_j = neighbor[do][1] + j
#         if 0 <= next_i < row and 0 <= next_j < col and (
#                 data[next_i][next_j] == '0'):
#             data[i][j] = '0'
#             data[neighbor[do][0] + i][neighbor[do][1] + j] = ind
#         else:
#             return
#     return data


# if __name__ == "__main__":
#     # 读取第一行的n
#     n, k = list(map(int, sys.stdin.readline().strip().split()))
#     data0 = []
#     for i in range(n):
#         # 读取每一行
#         line = sys.stdin.readline().strip()
#         # print(line)
#         # 把每一行的数字分隔后转化成int列表
#         values = list(line)
#         # print(values)
#         data0.append(values)
#     print(data0)
#     for j in range(k):
#         line = sys.stdin.readline().strip().split()
#         ind = line[0]
#         do = line[1]
#         # print(do)
#         res = mapMoving(data0, ind, do)
#         # print(res)
#         if res:
#             data0 = res
#         # print(data0)

#         for i in range(len(data0)):
#             for j in range(len(data0[0])):
#                 print(data0[i][j], end='')
#             print('')

# # 8 1
# # 00000000
# # 10044000
# # 10040000
# # 10040000
# # 00440220
# # 00040020
# # 00044030
# # 00000000
# # 1 L
# # 2 D
# # 3 R
# # 1 U
print('hello world')