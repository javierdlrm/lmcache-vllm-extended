ssh: 
	@echo "sshing into VM: password is FsM9PT7oDCVD"
	ssh -J ik2221vt25g09@nslabgw.it.kth.se ik2221vt25g09@192.168.2.27
# password 

scp:
	@echo "copying project to VM: password is FsM9PT7oDCVD"
	scp -J ik2221vt25g09@nslabgw.it.kth.se  -r "$(shell pwd)" ik2221vt25g09@192.168.2.27:/home/ik2221vt25g09/lmcache-vllm-extended

scpfrontend:
	@echo "copying frontend to VM: password is FsM9PT7oDCVD"
	scp -J ik2221vt25g09@nslabgw.it.kth.se  -r "$(shell pwd)/frontend" ik2221vt25g09@192.168.2.27:/home/ik2221vt25g09/lmcache-vllm-extended

scpdata:
	@echo "copying frontend/data to VM: password is FsM9PT7oDCVD"
	scp -J ik2221vt25g09@nslabgw.it.kth.se  -r "$(shell pwd)/frontend/data" ik2221vt25g09@192.168.2.27:/home/ik2221vt25g09/lmcache-vllm-extended/frontend

scpprompts:
	@echo "copying frontend/prompts to VM: password is FsM9PT7oDCVD"
	scp -J ik2221vt25g09@nslabgw.it.kth.se  -r "$(shell pwd)/frontend/prompts" ik2221vt25g09@192.168.2.27:/home/ik2221vt25g09/lmcache-vllm-extended/frontend

portforward:
	@echo "port forwarding to VM: password is FsM9PT7oDCVD"
	ssh -J ik2221vt25g09@nslabgw.it.kth.se ik2221vt25g09@192.168.2.27 -L 8501:localhost:8501 -N



# copy-logs:
# 	@echo "copying logs from tmp"
# 	-cp /tmp/*.stdout ./logs
# 	-cp /tmp/*.stderr ./logs
# 	-cp /tmp/*.report ./results

# get-logs:
# 	@echo "copying logs from VM"
# 	-scp -r -P 2222 ik2221@localhost:/home/ik2221/ik2221-assign-phase1-team3/logs/*.{stdout,stderr} ./logs
# 	-scp -r -P 2222 ik2221@localhost:/home/ik2221/ik2221-assign-phase1-team3/results/*.report ./results
# 	-scp -r -P 2222 ik2221@localhost:/home/ik2221/ik2221-assign-phase1-team3/results/phase_1_report ./