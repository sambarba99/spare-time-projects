Critical path analysis demo (example: engineering project)

| Task description | Code | Duration | Predecessors |
| :---: | :---: | :---: | :---: |
| Analysis | 0 | 120 | None |
| Design | 1 | 60 | 0 |
| Layout | 2 | 15 | 0 |
| Request material | 3 | 3 | 1,2 |
| Request parts | 4 | 3 | 1,2 |
| Receive material | 5 | 7 | 3 |
| Receive parts | 6 | 7 | 4 |
| Fabrication | 7 | 25 | 2,5 |
| Assembly | 8 | 60 | 2,6,7 |
| Testing | 9 | 90 | 8 |

Output:

<p align="center">
	<img src="cpa_output.png"/>
</p>
